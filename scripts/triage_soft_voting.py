import pandas as pd
import numpy as np
import joblib
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
from catboost import CatBoostClassifier
from mlxtend.evaluate import bias_variance_decomp

# I reduced the dataset to 40% of the original size and saved it as 'subset.csv'

#df=pd.read_csv('1stproject.csv')
#df_subset= df.sample(frac=0.4,random_state=1)
#df_subset.to_csv('subset.csv',index=False) 

df = pd.read_csv('data/subset.csv')
X = df.drop('disposition', axis=1) 
y = df['disposition']

# Split the data first to avoid leakage in encoding and imputation
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

# Then i transformed my target value into 0/1 where 0 is "Admit" and 1 is "Discharge"
label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train_raw)
y_test = label_enc.transform(y_test_raw)

# Also i transformed every feature into numbers - Handling categorical columns properly after split
X_train_numeric = pd.get_dummies(X_train_raw)
X_test_numeric = pd.get_dummies(X_test_raw)
# Ensure both sets have the same columns after get_dummies
X_test_numeric = X_test_numeric.reindex(columns=X_train_numeric.columns, fill_value=0)

# Handled missing values using median imputation based only on training data
train_median = X_train_numeric.median()
X_train_imputed = X_train_numeric.fillna(train_median)
X_test_imputed = X_test_numeric.fillna(train_median)

 # Feature Selection: Apply Variance Threshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train_imputed)
X_train_sel = pd.DataFrame(selector.transform(X_train_imputed), columns=X_train_numeric.columns[selector.get_support()])
X_test_sel = pd.DataFrame(selector.transform(X_test_imputed), columns=X_train_numeric.columns[selector.get_support()])

# Performed a Random forest in order to get the top 150 features
rf_selector = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_sel, y_train)
importances = pd.Series(rf_selector.feature_importances_, index=X_train_sel.columns)
top_features = importances.nlargest(150).index.tolist()

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
        ('clf', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1,num_leaves=31,reg_lambda=10)) 
    ])),  
    ('meds_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', meds_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1, n_jobs=-1))
    ])), 
    ('labs_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', labs_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(iterations=300, verbose=0,learning_rate=0.05,depth=8 ,thread_count=-1))
    ])), 
    ('history_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', history_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', CatBoostClassifier(iterations=300, verbose=0,learning_rate=0.05,depth=8,thread_count=-1))
    ]))
]

#  Hyperparameter grid for RandomizedSearchCV 
param_distributions = {'weights': np.random.uniform(0.5, 2.0, (10, 4)).tolist()}

# Initialize the Voting Classifier with 'soft' voting
Voting_model = VotingClassifier(estimators=base_learners, voting='soft')

# Create Custom Scorer for the admission class
f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=0)
# Perform Randomized Search Cross-Validation (3-fold) to find the best hyperparameter combination.
search = RandomizedSearchCV(estimator=Voting_model, param_distributions=param_distributions, 
                            n_iter=7, cv=3, scoring=f2_scorer, verbose=1, n_jobs=-1, random_state=42)
search.fit(X_train.values, y_train) 

# Final model
final_model = search.best_estimator_

# Bias-Variance Decomposition (mlxtend)
sample_size = min(5000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train.values[indices]
y_sample = y_train[indices]

mse, bias, var = bias_variance_decomp(
    final_model, 
    X_sample, y_sample, 
    X_test.values, y_test, 
    loss='0-1_loss', num_rounds=10, random_seed=42
)

# Evaluation 

# Extract probabilities for the Admit class (index 0) from the search result
y_prob_train_admit = final_model.predict_proba(X_train.values)[:, 0]
y_prob_test_admit = final_model.predict_proba(X_test.values)[:, 0]

# Elbow method to find optimal threshold
precision_tr, recall_tr, thresholds_tr = precision_recall_curve(y_train, y_prob_train_admit, pos_label=0)
distances = np.sqrt((1 - precision_tr)**2 + (1 - recall_tr)**2)
best_threshold = thresholds_tr[np.argmin(distances)]

# Final predictions using the elbow-optimized threshold
y_pred_custom = np.where(y_prob_test_admit > best_threshold, 0, 1)

# Metrics for reporting
f2_admit = fbeta_score(y_test, y_pred_custom, beta=2, pos_label=0)
report = classification_report(y_test, y_pred_custom, target_names=label_enc.classes_)
cm = confusion_matrix(y_test, y_pred_custom)

# Learning Curve using the best estimator
train_sizes, train_scores, val_scores = learning_curve(
    estimator=final_model, X=X_train.values, y=y_train, train_sizes=np.linspace(0.1, 1.0, 5),
    cv=3, scoring=f2_scorer, n_jobs=-1, random_state=42
)
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

# Dashboard Setup (Removed Calibration Plot)
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Soft-Voting Evaluation Dashboard', fontsize=22, fontweight='bold', y=0.98)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_test_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend()

# PR Curve with Elbow
axes[0, 1].plot(recall_tr, precision_tr, color='red', label='PR Curve (Train)')
idx = np.argmin(distances)
axes[0, 1].scatter(recall_tr[idx], precision_tr[idx], color='black', s=100, label=f'Elbow (t={best_threshold:.2f})', zorder=5)
axes[0, 1].set_title('Precision-Recall Curve (Elbow)'); axes[0, 1].legend()

# Expert Performance
expert_names = [name.replace('_expert', '') for name, _ in final_model.estimators]
expert_aucs = [roc_auc_score(y_test, final_model.estimators_[i].predict_proba(X_test.values)[:, 1]) for i in range(len(expert_names))]
axes[0, 2].bar(expert_names, expert_aucs, color='skyblue')
axes[0, 2].set_title('AUC per Expert')

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_).plot(ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('Confusion Matrix')

# Learning Curve
axes[1, 1].plot(train_sizes, train_mean, label='Train'); axes[1, 1].plot(train_sizes, val_mean, label='Val')
axes[1, 1].set_title('Learning Curve'); axes[1, 1].legend()

# Bias-Variance Bar Chart
axes[1, 2].bar(['Bias', 'Variance', 'MSE'], [bias, var, mse], color=['blue', 'red', 'green'], alpha=0.7)
axes[1, 2].set_title('Bias-Variance Decomposition')
for i, v in enumerate([bias, var, mse]): axes[1, 2].text(i, v + 0.005, f'{v:.3f}', ha='center')

# Metrics Text 
plt.figtext(0.85, 0.25, f"Elbow Threshold: {best_threshold:.2f}\nF2-Score: {f2_admit:.4f}\n\n{report}", 
            fontsize=10, family='monospace', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

# Save model artifacts for future use
model_artifacts = {
    'model': final_model,             
    'label_encoder': label_enc,
    'train_median': train_median,
    'selector': selector,              
    'top_features': top_features,
    'best_threshold': best_threshold
}

joblib.dump(model_artifacts, 'models/soft_voting_artifacts.pkl')

plt.savefig('plots/soft_voting_eval.png', dpi=300, bbox_inches='tight')
plt.show()