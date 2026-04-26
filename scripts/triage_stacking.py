import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, fbeta_score, 
                             classification_report, confusion_matrix, roc_auc_score, 
                             make_scorer, ConfusionMatrixDisplay)
from catboost import CatBoostClassifier
from mlxtend.evaluate import bias_variance_decomp

# 1. Load data and initialize target encoding
df = pd.read_csv('data/subset.csv')
X = df.drop('disposition', axis=1)
y = df['disposition']

# 2. Split data with stratification
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train_raw)
y_test = label_enc.transform(y_test_raw)

X_train, X_test = X_train.copy(), X_test.copy()

# 3. Handle missing values
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

train_median = X_train[num_cols].median()
X_train[num_cols] = X_train[num_cols].fillna(train_median)
X_test[num_cols] = X_test[num_cols].fillna(train_median)
X_train[cat_cols] = X_train[cat_cols].fillna('Missing')
X_test[cat_cols] = X_test[cat_cols].fillna('Missing')

# 4. One-Hot Encoding and column name cleaning
X_train_dum = pd.get_dummies(X_train)
X_test_dum = pd.get_dummies(X_test)
X_train_dum, X_test_dum = X_train_dum.align(X_test_dum, join='left', axis=1, fill_value=0)

bool_cols = X_train_dum.select_dtypes(include=['uint8', 'bool']).columns
X_train_dum[bool_cols] = X_train_dum[bool_cols].astype('int8')
X_test_dum[bool_cols] = X_test_dum[bool_cols].astype('int8')

X_train_dum.columns = X_train_dum.columns.str.replace(r'[\[\]<>,:{}\"]', '_', regex=True)
X_test_dum.columns = X_test_dum.columns.str.replace(r'[\[\]<>,:{}\"]', '_', regex=True)

# 5. Feature Selection
X_fs, X_train_cv, y_fs, y_train_cv = train_test_split(
    X_train_dum, y_train, test_size=0.8, random_state=42, stratify=y_train
)
lgb_selector = lgb.LGBMClassifier(n_estimators=100, importance_type='gain', n_jobs=-1, random_state=42)
lgb_selector.fit(X_fs, y_fs)
top_100_features = pd.Series(lgb_selector.feature_importances_, index=X_train_dum.columns).nlargest(100).index.tolist()
X_train_sel = X_train_dum[top_100_features].copy()
X_test_sel = X_test_dum[top_100_features].copy()

# 6. Feature Subspacing
random.seed(22669234)
top_5_anchors = top_100_features[:5]
remaining_95 = [f for f in top_100_features if f not in top_5_anchors]
random.shuffle(remaining_95)
b1, b2, b3, b4 = remaining_95[0:24], remaining_95[24:48], remaining_95[48:72], remaining_95[72:95]

def build_group(base, pool, target_size=55):
    needed = target_size - len(base)
    pool_filtered = [f for f in pool if f not in base]
    extra = random.sample(pool_filtered, min(needed, len(pool_filtered)))
    return base + extra

g1, g2, g3, g4 = [top_5_anchors + build_group(b, remaining_95) for b in [b1, b2, b3, b4]]

def get_indices(df, col_list):
    return [df.columns.get_loc(c) for c in col_list]

idx1, idx2, idx3, idx4 = [get_indices(X_train_sel, g) for g in [g1, g2, g3, g4]]

# 7. Model Stacking
ratio = np.bincount(y_train)[1] / np.bincount(y_train)[0]
base_learners = [
    ('sub1_lgbm', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx1)], remainder='drop')), 
                             ('clf', lgb.LGBMClassifier(n_estimators=100, scale_pos_weight=ratio, random_state=42, n_jobs=-1))])),
    ('sub2_rf', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx2)], remainder='drop')), 
                            ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=5, min_samples_leaf=50, random_state=42, n_jobs=-1))])),
    ('sub3_cat', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx3)], remainder='drop')), 
                             ('clf', CatBoostClassifier(iterations=200, verbose=0, auto_class_weights='Balanced', depth=4, thread_count=-1))])),
    ('sub4_cat_alt', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx4)], remainder='drop')), 
                                 ('clf', CatBoostClassifier(iterations=200, verbose=0, depth=5, thread_count=-1))]))
]
meta_model = LogisticRegression(max_iter=1000, random_state=42, C=0.01, n_jobs=-1,class_weight='balanced')
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=3, stack_method='predict_proba', n_jobs=-1)

# Training
print("Training Stacking model...")
stacking_model.fit(X_train_sel.values, y_train)

# 8. Evaluation & Threshold (Elbow)
y_prob_train_oof = cross_val_predict(stacking_model, X_train_sel.values, y_train, cv=3, method='predict_proba', n_jobs=-1)[:, 0]
prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_train, y_prob_train_oof, pos_label=0)
distances = np.sqrt((1 - prec_tr)**2 + (1 - rec_tr)**2)
best_t = thresh_tr[np.argmin(distances)]

y_prob_admit = stacking_model.predict_proba(X_test_sel.values)[:, 0]
y_pred = np.where(y_prob_admit > best_t, 0, 1)

# Bias-Variance
mse, bias, var = bias_variance_decomp(stacking_model, X_train_sel.values[:1500], y_train[:1500], X_test_sel.values, y_test, loss='0-1_loss', num_rounds=5, random_seed=42)

# 9. DASHBOARD 
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
fig.suptitle('Stacking Evaluation Dashboard', fontsize=22, fontweight='bold', y=0.98)

fpr, tpr, _ = roc_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend()

axes[0, 1].plot(rec_tr, prec_tr, color='red', label='PR Curve (Train)')
idx_e = np.argmin(distances)
axes[0, 1].scatter(rec_tr[idx_e], prec_tr[idx_e], color='black', s=100, label=f'Elbow (t={best_t:.2f})', zorder=5)
axes[0, 1].set_title('Precision-Recall Curve (Elbow)'); axes[0, 1].legend()

expert_aucs = []
for name, est in stacking_model.named_estimators_.items():
    expert_aucs.append(roc_auc_score(y_test, est.predict_proba(X_test_sel.values)[:, 1]))
axes[0, 2].bar(['g1', 'g2', 'g3', 'g4'], expert_aucs, color='skyblue')
axes[0, 2].set_title('AUC per Expert')

axes[0, 3].axis('off')

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=label_enc.classes_).plot(ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('Confusion Matrix')

f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=0)
ts, tr_s, vl_s = learning_curve(stacking_model, X_train_sel.values, y_train, train_sizes=np.linspace(0.1, 1.0, 5), cv=2, scoring=f2_scorer)
axes[1, 1].plot(ts, np.mean(tr_s, axis=1), label='Train'); axes[1, 1].plot(ts, np.mean(vl_s, axis=1), label='Val')
axes[1, 1].set_title('Learning Curve'); axes[1, 1].legend()

axes[1, 2].bar(['Bias', 'Variance', 'MSE'], [bias, var, mse], color=['blue', 'red', 'green'], alpha=0.7)
axes[1, 2].set_title('Bias-Variance Decomposition')
for i, v in enumerate([bias, var, mse]): axes[1, 2].text(i, v + 0.005, f'{v:.3f}', ha='center')

axes[1, 3].axis('off')
report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
metrics_text = f"Elbow Threshold: {best_t:.2f}\nF2-Score: {fbeta_score(y_test, y_pred, beta=2, pos_label=0):.4f}\n\n{report}"
axes[1, 3].text(-0.1, 1.0, metrics_text, fontsize=10, family='monospace', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

model_artifacts = {
    'model': stacking_model,
    'label_encoder': label_enc,
    'train_median': train_median,
    'top_100_features': top_100_features,
    'best_threshold': best_t
}

joblib.dump(model_artifacts, 'models/stacking_model_artifacts.pkl')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/stacking_eval.png', dpi=300, bbox_inches='tight')
plt.show()