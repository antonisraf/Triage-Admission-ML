import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, fbeta_score, 
                             classification_report, confusion_matrix, roc_auc_score, 
                             make_scorer, ConfusionMatrixDisplay)
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from catboost import CatBoostClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import cross_val_predict

# 1. Load data and initialize target encoding
df = pd.read_csv('data/subset.csv')
X = df.drop('disposition', axis=1)
y = df['disposition']

label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# 2. Split data with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
X_train, X_test = X_train.copy(), X_test.copy()

# Calculate class ratio for balancing base learners
ratio = np.bincount(y_train)[1] / np.bincount(y_train)[0]

# 3. Handle missing values: Median for numerical, 'Missing' for categorical
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

train_median = X_train[num_cols].median()
X_train[num_cols] = X_train[num_cols].fillna(train_median)
X_test[num_cols] = X_test[num_cols].fillna(train_median)
X_train[cat_cols] = X_train[cat_cols].fillna('Missing')
X_test[cat_cols] = X_test[cat_cols].fillna('Missing')

# 4. One-Hot Encoding and column name cleaning for LightGBM compatibility
X_train_dum, X_test_dum = pd.get_dummies(X_train).align(pd.get_dummies(X_test), join='left', axis=1, fill_value=0)

bool_cols = X_train_dum.select_dtypes(include=['uint8', 'bool']).columns
X_train_dum[bool_cols] = X_train_dum[bool_cols].astype('int8')
X_test_dum[bool_cols] = X_test_dum[bool_cols].astype('int8')

X_train_dum.columns = X_train_dum.columns.str.replace(r'[\[\]<>,:{}\"]', '_', regex=True)
X_test_dum.columns = X_test_dum.columns.str.replace(r'[\[\]<>,:{}\"]', '_', regex=True)

# 5. Feature Selection: Keep top 100 features based on LightGBM gain
lgb_selector = lgb.LGBMClassifier(n_estimators=200, importance_type='gain', n_jobs=-1)
lgb_selector.fit(X_train_dum, y_train)

top_100_features = pd.Series(lgb_selector.feature_importances_, index=X_train_dum.columns).nlargest(100).index.tolist()
X_train_sel = X_train_dum[top_100_features].copy()
X_test_sel = X_test_dum[top_100_features].copy()

# 6. Feature Subspacing: Create 4 groups of 60 features using 5 anchors and random sampling
random.seed(22669234)
top_5_anchors = top_100_features[:5]
remaining_95 = [f for f in top_100_features if f not in top_5_anchors]
random.shuffle(remaining_95)

b1, b2, b3, b4 = remaining_95[0:24], remaining_95[24:48], remaining_95[48:72], remaining_95[72:95]

def build_group(base, pool, target_size=55):
    needed = target_size - len(base)
    extra = random.sample([f for f in pool if f not in base], needed)
    return base + extra

g1, g2, g3, g4 = [top_5_anchors + build_group(b, remaining_95) for b in [b1, b2, b3, b4]]

def get_indices(df, col_list):
    return [df.columns.get_loc(c) for c in col_list]

idx1, idx2, idx3, idx4 = [get_indices(X_train_sel, g) for g in [g1, g2, g3, g4]]

# 7. Model Stacking: Define heterogeneous base learners and an LGBM meta-learner
base_learners = [
    ('sub1_lgbm', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx1)], remainder='drop')), 
                             ('clf', lgb.LGBMClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, n_jobs=-1))])),
    ('sub2_rf', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx2)], remainder='drop')), 
                            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=8, random_state=42, n_jobs=-1))])),
    ('sub3_cat', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx3)], remainder='drop')), 
                             ('clf', CatBoostClassifier(iterations=200, verbose=0, auto_class_weights='Balanced', depth=4, thread_count=-1))])),
    ('sub4_cat_alt', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx4)], remainder='drop')), 
                                 ('clf', CatBoostClassifier(iterations=200, verbose=0, depth=5, thread_count=-1))]))
]

meta_model = lgb.LGBMClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42, n_jobs=-1)

stacking_base = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=3, stack_method='predict_proba', n_jobs=-1)

# 8. Probability Calibration: Use isotonic regression for better reliability
print("Training Calibrated Stacking...")
calibrated_stacking = CalibratedClassifierCV(stacking_base, method='isotonic', cv=3)
calibrated_stacking.fit(X_train_sel.values, y_train)

# 9. Performance Evaluation: Optimized threshold search for F-beta score
y_prob_train_oof = cross_val_predict(calibrated_stacking, X_train_sel.values, y_train, cv=3, method='predict_proba', n_jobs=-1)[:, 0]

thresholds = np.linspace(0, 1, 100)
f2_scores = [fbeta_score(y_train, np.where(y_prob_train_oof > t, 0, 1), beta=1.2, pos_label=0) for t in thresholds]
best_t = thresholds[np.argmax(f2_scores)]

y_prob_admit = calibrated_stacking.predict_proba(X_test_sel.values)[:, 0]
y_pred = np.where(y_prob_admit > best_t, 0, 1)

# Bias-Variance decomposition analysis
mse, bias, var = bias_variance_decomp(calibrated_stacking, X_train_sel.values[:1500], y_train[:1500], X_test_sel.values, y_test, loss='0-1_loss', num_rounds=5, random_seed=42)

# 10. Dashboard Visualization
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
fig.suptitle('Stacking Evaluation Dashboard', fontsize=20, fontweight='bold')

# ROC Curve plot
fpr, tpr, _ = roc_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.2f}'); axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend()

# Precision-Recall Curve plot
prec, rec, _ = precision_recall_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 1].plot(rec, prec, color='red', label=f'AUC={auc(rec, prec):.3f}'); axes[0, 1].set_title('PR Curve'); axes[0, 1].legend()

# Individual subspace performance (AUC per Base Learner)
stacking_inside = calibrated_stacking.calibrated_classifiers_[0].estimator
expert_names, expert_aucs = [], []
for name, est in stacking_inside.named_estimators_.items():
    expert_aucs.append(roc_auc_score(y_test, est.predict_proba(X_test_sel.values)[:, 1]))
    expert_names.append(name)
axes[0, 2].bar(expert_names, expert_aucs, color='skyblue'); axes[0, 2].set_title('AUC per Subspace'); axes[0, 2].tick_params(axis='x', rotation=45)

# Confusion Matrix plot
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=label_enc.classes_).plot(ax=axes[0, 3], cmap='Blues')

# Probability Calibration and Learning Curve plots
CalibrationDisplay.from_predictions((y_test==0).astype(int), y_prob_admit, n_bins=10, ax=axes[1, 0])

f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=0)
ts, tr_s, vl_s = learning_curve(calibrated_stacking, X_train_sel.values, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=2, scoring=f2_scorer)
axes[1, 1].plot(ts, np.mean(tr_s, axis=1), label='Train'); axes[1, 1].plot(ts, np.mean(vl_s, axis=1), label='Val'); axes[1, 1].legend()

# Bias-Variance decomposition bar chart
axes[1, 2].bar(['Bias', 'Variance', 'MSE'], [bias, var, mse], color=['blue', 'red', 'green'])

# Final classification report and threshold metrics
axes[1, 3].axis('off')
axes[1, 3].text(0, 1, f"Best Threshold: {best_t:.2f}\nF2-Score: {max(f2_scores):.4f}\n\n{classification_report(y_test, y_pred)}", fontsize=10, family='monospace', va='top')

# save the model artifacts for future use
model_artifacts = {
    'model': calibrated_stacking,
    'label_encoder': label_enc,
    'train_median': train_median,
    'top_100_features': top_100_features,
    'best_threshold': best_t
}

joblib.dump(model_artifacts, 'models/stacking_model_artifacts.pkl')

# display the dashboard
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/stacking_eval.png', dpi=300, bbox_inches='tight')
plt.show()

