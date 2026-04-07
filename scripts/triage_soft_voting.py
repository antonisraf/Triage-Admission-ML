import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, learning_curve
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, fbeta_score, 
                             classification_report, confusion_matrix, roc_auc_score,make_scorer,
                             ConfusionMatrixDisplay)
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from catboost import CatBoostClassifier


# I reduced the dataset to 40% of the original size and saved it as 'subset.csv'

#df=pd.read_csv('1stproject.csv')
#df_subset= df.sample(frac=0.4,random_state=1)
#df_subset.to_csv('subset.csv',index=False) 

df = pd.read_csv('../data/subset.csv')
X = df.drop('disposition', axis=1) 
y = df['disposition']

# Then i transformed my target value into 0/1 where 0 is "Admit" and 1 is "Discharge"
# Also i transformed every feature into numbers
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)
df_numeric = pd.get_dummies(X)

# Split the data into 80% training and 20% test 
# I used stratification in order to have balanced classes for the train/test set
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df_numeric, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Handled missing values using median imputation 
train_median = X_train_raw.median()
X_train_imputed = X_train_raw.fillna(train_median)
X_test_imputed = X_test_raw.fillna(train_median)

 # Feature Selection: Apply Variance Threshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train_imputed)
X_train_sel = pd.DataFrame(selector.transform(X_train_imputed), columns=X_train_raw.columns[selector.get_support()])
X_test_sel = pd.DataFrame(selector.transform(X_test_imputed), columns=X_train_raw.columns[selector.get_support()])

# Performed a Random forest in order to get the top 100 features
rf_selector = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_sel, y_train)
importances = pd.Series(rf_selector.feature_importances_, index=X_train_sel.columns)
top_features = importances.nlargest(100).index.tolist()

# Split the features into 4 groups based on keywords
vitals_top = [c for c in top_features if any(w in c.lower() for w in ['vital', 'sbp', 'dbp', 'pulse', 'temp', 'o2', 'hr', 'rr'])]
meds_top = [c for c in top_features if c.startswith('meds_') and c not in vitals_top]
labs_top = [c for c in top_features if any(w in c.lower() for w in ['median', 'min', 'max', 'last']) and c not in vitals_top and c not in meds_top]
history_top = [c for c in top_features if c not in vitals_top + meds_top + labs_top]


X_train = X_train_sel[top_features]
X_test = X_test_sel[top_features]

# 4 Base learners for each team of features
base_learners = [
    ('vitals_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', vitals_top)], remainder='drop')),
        ('scaler', StandardScaler()), 
        ('clf', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, class_weight='balanced',reg_lambda=10, n_jobs=-1, verbosity=-1)) 
    ])),  
    ('meds_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', meds_top)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=0.01, n_jobs=-1))
    ])), 
    ('labs_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', labs_top)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced_subsample', max_depth=8, min_samples_leaf=50, max_features=0.5, random_state=42, n_jobs=-1))
    ])), 
    ('history_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', history_top)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0, thread_count=-1))
    ]))
]


# Hyperparameter grid for RandomizedSearchCV 
param_distributions = {
    
    'meds_expert__clf__C': [0.01, 0.05, 0.1],
    'labs_expert__clf__max_depth': [5, 8, 10], 
    'weights': [[1.5, 1, 0.5, 1.5], [2, 1, 1, 2], [1, 1, 0.8, 1.2], [1.2, 0.8, 0.5, 2]]
}
    

# Initialize the Voting Classifier with 'soft' voting
Voting_model = VotingClassifier(estimators=base_learners, voting='soft')

# Create Custom Scorer for the admission class
f2_scorer= make_scorer (fbeta_score,beta=2,pos_label=0)

# Perform Randomized Search Cross-Validation (5-fold) to find the best hyperparameter combination.
search = RandomizedSearchCV(estimator=Voting_model, param_distributions=param_distributions, 
                            n_iter=15, cv=5, scoring=f2_scorer, verbose=1, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

# Propability Calibration applied isotonic regression to calibrate the predicted propabilities
calibrated_model = CalibratedClassifierCV(search.best_estimator_, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# Evaluation

# Extract probalities for the Admit class
y_prob_admit = calibrated_model.predict_proba(X_test)[:, 0]
# Sensitivity threshold to prioritize admissions
y_pred_custom = np.where(y_prob_admit > 0.35, 0, 1)
y_test_admit = (y_test == 0).astype(int)

# f2 score
f2_admit = fbeta_score(y_test, y_pred_custom, beta=2, pos_label=0)
# Classification report
report = classification_report(y_test, y_pred_custom, target_names=label_enc.classes_)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_custom)

# Learning Curve 
train_sizes, train_scores, val_scores = learning_curve(
    estimator=calibrated_model, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3, scoring=f2_scorer, n_jobs=-1, random_state=42
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# Dashboard Setup
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
fig.suptitle('Extended Evaluation Dashboard (Calibrated Voting Classifier)', fontsize=22, fontweight='bold', y=0.98)

fpr, tpr, _ = roc_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
axes[0, 0].plot([0, 1], [0, 1], linestyle='--', color='gray'); axes[0, 0].set_title('ROC Curve (Admit)'); axes[0, 0].legend(loc='lower right')

precision, recall, _ = precision_recall_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 1].plot(recall, precision, color='red', label=f'AUC = {auc(recall, precision):.3f}')
axes[0, 1].set_title('Precision-Recall Curve'); axes[0, 1].legend()

expert_names, expert_aucs = [], []
for i, (name, _) in enumerate(calibrated_model.estimator.estimators):
    expert_model = calibrated_model.estimator.estimators_[i]
    prob = expert_model.predict_proba(X_test)[:, 1]
    expert_names.append(name.replace('_expert', ''))
    expert_aucs.append(roc_auc_score(y_test, prob))
axes[0, 2].bar(expert_names, expert_aucs, color='skyblue', edgecolor='black')
axes[0, 2].set_title('General AUC per Expert'); axes[0, 2].set_ylim(0, 1.1)
for i, v in enumerate(expert_aucs): axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_).plot(ax=axes[0, 3], cmap='Blues', colorbar=False,values_format='d')
axes[0, 3].set_title('Confusion Matrix')

CalibrationDisplay.from_predictions(y_test_admit, y_prob_admit, n_bins=10, name='Calibrated', ax=axes[1, 0])
axes[1, 0].set_title('Calibration Curve')

axes[1, 1].plot(train_sizes, train_mean, 'o-', label='Training Score')
axes[1, 1].plot(train_sizes, val_mean, 'o-', label='Cross-validation Score')
axes[1, 1].set_title('Learning Curve'); axes[1, 1].legend()

x_bv = np.linspace(0.5, 5, 100)
axes[1, 2].plot(x_bv, 1/x_bv**2, label='Bias²', color='blue'); axes[1, 2].plot(x_bv, 0.1*x_bv**1.5, label='Variance', color='red')
axes[1, 2].plot(x_bv, (1/x_bv**2)+(0.1*x_bv**1.5), label='Total Error', color='green', linewidth=2)
axes[1, 2].set_title('Bias-Variance Tradeoff'); axes[1, 2].legend(); axes[1, 2].get_xaxis().set_ticks([])

axes[1, 3].axis('off')
text_str = f"--- Basic Metrics ---\n\nF2-Score: {f2_admit:.4f}\n\n{report}"
axes[1, 3].text(-0.15, 1.05, text_str, fontsize=9, family='monospace', va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.6', facecolor='whitesmoke', alpha=0.9))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the dashboard 
plt.savefig('../plots/soft_voting_eval.png', dpi=300, bbox_inches='tight')
# Show the dashboard
plt.show()
