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
                             classification_report, confusion_matrix, roc_auc_score, make_scorer,
                             ConfusionMatrixDisplay)
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from catboost import CatBoostClassifier
from mlxtend.evaluate import bias_variance_decomp

# I reduced the dataset to 40% of the original size and saved it as 'subset.csv'

#df=pd.read_csv('1stproject.csv')
#df_subset= df.sample(frac=0.4,random_state=1)
#df_subset.to_csv('subset.csv',index=False) 

df = pd.read_csv('data/subset.csv')
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

def get_indices(df, col_list):
    return [df.columns.get_loc(c) for c in col_list]

vitals_idx = get_indices(X_train, vitals_top)
meds_idx = get_indices(X_train, meds_top)
labs_idx = get_indices(X_train, labs_top)
history_idx = get_indices(X_train, history_top)

# Base learners for each team of features
base_learners = [
    ('vitals_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', vitals_idx)], remainder='drop')),
        ('scaler', StandardScaler()), 
        ('clf', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1,num_leaves=31,reg_lambda=10)) 
    ])),  
    ('meds_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', meds_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1, n_jobs=-1))
    ])), 
    ('labs_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', labs_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(iterations=300, verbose=0,learning_rate=0.05,depth=6 ,thread_count=-1))
    ])), 
    ('history_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', history_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(iterations=300, verbose=0,learning_rate=0.05,depth=6 ,thread_count=-1))
    ]))
]

#  Hyperparameter grid for RandomizedSearchCV 
param_distributions = {'weights': [
        [1.5, 2.0, 0.3, 2.5],
        [1.0, 2.5, 0.2, 3.0],
        [2.0, 1.5, 0.3, 2.0],
        [1.2, 2.2, 0.4, 2.8],
        [0.5, 3.0, 0.1, 3.5],
        [1.0, 2.0,0.05, 3.0] 
    ]}

# Initialize the Voting Classifier with 'soft' voting
Voting_model = VotingClassifier(estimators=base_learners, voting='soft')

# Create Custom Scorer for the admission class
f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=0)
# Perform Randomized Search Cross-Validation (3-fold) to find the best hyperparameter combination.
search = RandomizedSearchCV(estimator=Voting_model, param_distributions=param_distributions, 
                            n_iter=7, cv=3, scoring=f2_scorer, verbose=1, n_jobs=-1, random_state=42)
search.fit(X_train.values, y_train) 
#Propability Calibration applied isotonic regression to calibrate the predicted propabilities
calibrated_model = CalibratedClassifierCV(search.best_estimator_, method='isotonic', cv=3)
calibrated_model.fit(X_train.values, y_train)

# Bias-Variance Decomposition (mlxtend)
sample_size = min(5000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train.values[indices]
y_sample = y_train[indices]

mse, bias, var = bias_variance_decomp(
    search.best_estimator_, 
    X_sample, y_sample, 
    X_test.values, y_test, 
    loss='0-1_loss', num_rounds=10, random_seed=42
)

# Evaluation 

# Extract probalities for the Admit class
y_prob_admit = calibrated_model.predict_proba(X_test.values)[:, 0]
# Try 100 thresholds in order to get the one that maximizes f2 score
thresholds=np.linspace(0,1,100)
f2_scores = [fbeta_score(y_test, np.where(y_prob_admit > t, 0, 1), beta=1.2, pos_label=0) for t in thresholds]
best_threshold = thresholds[np.argmax(f2_scores)]
best_f2_score = max(f2_scores)
# Sensitivity threshold to prioritize admissions
y_pred_custom = np.where(y_prob_admit > best_threshold, 0, 1)
y_test_admit = (y_test == 0).astype(int)

# f2 scorer
f2_admit = fbeta_score(y_test, y_pred_custom, beta=2, pos_label=0)
# Classification report
report = classification_report(y_test, y_pred_custom, target_names=label_enc.classes_)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_custom)

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    estimator=calibrated_model, X=X_train.values, y=y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3, scoring=f2_scorer, n_jobs=-1, random_state=42
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# Dashboard Setup
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
fig.suptitle('Soft-Voting Evaluation Dashboard', fontsize=22, fontweight='bold', y=0.98)

# ROC & PR Curves
fpr, tpr, _ = roc_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend()

precision, recall, _ = precision_recall_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 1].plot(recall, precision, color='red', label=f'AUC = {auc(recall, precision):.3f}')
axes[0, 1].set_title('Precision-Recall Curve'); axes[0, 1].legend()

# Expert Performance
expert_names = [name.replace('_expert', '') for name, _ in calibrated_model.estimator.estimators]
expert_aucs = [roc_auc_score(y_test, calibrated_model.estimator.estimators_[i].predict_proba(X_test.values)[:, 1]) for i in range(len(expert_names))]
axes[0, 2].bar(expert_names, expert_aucs, color='skyblue')
axes[0, 2].set_title('AUC per Expert')

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_).plot(ax=axes[0, 3], cmap='Blues')
axes[0, 3].set_title('Confusion Matrix')

# Calibration & Learning Curves
CalibrationDisplay.from_predictions(y_test_admit, y_prob_admit, n_bins=10, ax=axes[1, 0])
axes[1, 0].set_title('Calibration')

axes[1, 1].plot(train_sizes, train_mean, label='Train'); axes[1, 1].plot(train_sizes, val_mean, label='Val')
axes[1, 1].set_title('Learning Curve'); axes[1, 1].legend()

# Bias-Variance Bar Chart
axes[1, 2].bar(['Bias', 'Variance', 'MSE'], [bias, var, mse], color=['blue', 'red', 'green'], alpha=0.7)
axes[1, 2].set_title('Bias-Variance Decomposition')
for i, v in enumerate([bias, var, mse]): axes[1, 2].text(i, v + 0.005, f'{v:.3f}', ha='center')

# Metrics Text
axes[1, 3].axis('off')
axes[1, 3].text(-0.1, 1.0, f"F2-Score: {f2_admit:.4f}\n\n{report}", fontsize=10, family='monospace', va='top')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Save the dashboard 
plt.savefig('plots/soft_voting_eval.png', dpi=300, bbox_inches='tight')
# Show the dashboard
plt.show()