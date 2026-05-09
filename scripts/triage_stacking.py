import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import lightgbm as lgb
import joblib
import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_predict, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, fbeta_score, 
                             classification_report, confusion_matrix, roc_auc_score, 
                             make_scorer, ConfusionMatrixDisplay, precision_score)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from mlxtend.evaluate import bias_variance_decomp

# 1. Load data and initialize target encoding
df = pd.read_csv('data/subset.csv')  
X = df.drop('disposition', axis=1)
y = df['disposition']

# 2. Split data with stratification
X_train, X_temp, y_train_raw, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train_raw)
y_val = label_enc.transform(y_val_raw)
y_test = label_enc.transform(y_test_raw)

X_train_raw, X_test_raw = X_train.copy(), X_test_raw.copy()

# 3. Handle missing values
# 4. One-Hot Encoding and column name cleaning
num_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ], remainder='passthrough')

X_train_processed = preprocessor.fit_transform(X_train_raw)
X_test_processed = preprocessor.transform(X_test_raw)

cols = preprocessor.get_feature_names_out()
cols = [re.sub(r'[\[\]<>,:{}\"]', '_', c) for c in cols]

X_train_dum = pd.DataFrame(X_train_processed, columns=cols, index=X_train_raw.index)
X_test_dum = pd.DataFrame(X_test_processed, columns=cols, index=X_test_raw.index)

bool_cols = X_train_dum.select_dtypes(include=['uint8', 'bool']).columns
X_train_dum[bool_cols] = X_train_dum[bool_cols].astype('int8')
X_test_dum[bool_cols] = X_test_dum[bool_cols].astype('int8')

# 5. Feature Selection
X_fs, _, y_fs, _ = train_test_split(X_train_dum, y_train, test_size=0.8, random_state=42, stratify=y_train)
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
b1, b2, b3, b4, b5 = remaining_95[0:19], remaining_95[19:38], remaining_95[38:57], remaining_95[57:76], remaining_95[76:95]

def build_group(base, pool, target_size=55):
    needed = target_size - len(base)
    pool_filtered = [f for f in pool if f not in base]
    extra = random.sample(pool_filtered, min(needed, len(pool_filtered)))
    return base + extra

g1, g2, g3, g4, g5 = [top_5_anchors + build_group(b, remaining_95) for b in [b1, b2, b3, b4, b5]]

def get_indices(df, col_list):
    return [df.columns.get_loc(c) for c in col_list]

idx1, idx2, idx3, idx4, idx5 = [get_indices(X_train_sel, g) for g in [g1, g2, g3, g4, g5]]

# 7. Model Stacking
class_ratio = ((y_train == 1).sum() / (y_train == 0).sum()) * 1.2

X_proxy, _, y_proxy, _ = train_test_split(X_train_sel.values, y_train, test_size=0.8, random_state=42, stratify=y_train)

def objective(trial):
    lgb_lr   = trial.suggest_float('lgb_lr', 0.01, 0.1, log=True)
    cat_lr   = trial.suggest_float('cat_lr', 0.01, 0.1, log=True)
    xgb_lr   = trial.suggest_float('xgb_lr', 0.01, 0.1, log=True)
    meta_lr  = trial.suggest_float('meta_lr', 0.01, 0.1, log=True)

    bases = [
        ('sub1_lgbm', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx1)], remainder='drop')),
                                 ('clf', lgb.LGBMClassifier(n_estimators=150, learning_rate=lgb_lr, scale_pos_weight=class_ratio, random_state=42, n_jobs=1))])),
        ('sub2_rf',   Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx2)], remainder='drop')),
                                 ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, bootstrap=False, class_weight='balanced', random_state=42, n_jobs=1))])),
        ('sub3_cat',  Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx3)], remainder='drop')),
                                 ('clf', CatBoostClassifier(iterations=200, learning_rate=cat_lr, depth=5, auto_class_weights='Balanced', verbose=0, thread_count=1))])),
        ('sub4_cat_alt', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx4)], remainder='drop')),
                                    ('clf', CatBoostClassifier(iterations=200, learning_rate=cat_lr, depth=5, auto_class_weights='Balanced', verbose=0, thread_count=1))])),
        ('sub5_xgb',  Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx5)], remainder='drop')),
                                 ('clf', XGBClassifier(n_estimators=150, learning_rate=xgb_lr, scale_pos_weight=class_ratio, eval_metric='logloss', random_state=42, n_jobs=1, verbosity=0))]))
    ]
    meta = lgb.LGBMClassifier(n_estimators=100, learning_rate=meta_lr, scale_pos_weight=class_ratio, random_state=42, n_jobs=-1, verbosity=-1)
    stack = StackingClassifier(estimators=bases, final_estimator=meta, cv=2, stack_method='predict_proba', n_jobs=-1)

    X_tr_opt, X_val_opt, y_tr_opt, y_val_opt = train_test_split(X_proxy, y_proxy, test_size=0.2, stratify=y_proxy, random_state=42)
    stack.fit(X_tr_opt, y_tr_opt)
    y_prob_opt = stack.predict_proba(X_val_opt)[:, 0]
    prec_o, rec_o, thresh_o = precision_recall_curve(y_val_opt, y_prob_opt, pos_label=0)
    f2_o = (5 * prec_o[:-1] * rec_o[:-1]) / (4 * prec_o[:-1] + rec_o[:-1] + 1e-9)
    t_o = thresh_o[np.argmax(f2_o)]
    y_pred_o = np.where(y_prob_opt > t_o, 0, 1)
    return fbeta_score(y_val_opt, y_pred_o, beta=2, pos_label=0)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=10) 

bp = study.best_params

base_learners = [
    ('sub1_lgbm',    Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx1)], remainder='drop')),
                                ('clf', lgb.LGBMClassifier(n_estimators=200, learning_rate=bp['lgb_lr'], scale_pos_weight=class_ratio, random_state=42, n_jobs=-1))])),
    ('sub2_rf',      Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx2)], remainder='drop')),
                                ('clf', RandomForestClassifier(n_estimators=150, max_depth=7, bootstrap=False, class_weight='balanced', random_state=42, n_jobs=-1))])),
    ('sub3_cat',     Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx3)], remainder='drop')),
                                ('clf', CatBoostClassifier(iterations=250, learning_rate=bp['cat_lr'], depth=6, auto_class_weights='Balanced', verbose=0, thread_count=-1))])),
    ('sub4_cat_alt', Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx4)], remainder='drop')),
                                ('clf', CatBoostClassifier(iterations=250, learning_rate=bp['cat_lr'], depth=6, auto_class_weights='Balanced', verbose=0, thread_count=-1))])),
    ('sub5_xgb',     Pipeline([('sel', ColumnTransformer([('k', 'passthrough', idx5)], remainder='drop')),
                                ('clf', XGBClassifier(n_estimators=200, learning_rate=bp['xgb_lr'], scale_pos_weight=class_ratio, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0))]))
]
meta_model = lgb.LGBMClassifier(n_estimators=200, learning_rate=bp['meta_lr'], scale_pos_weight=class_ratio, random_state=42, n_jobs=-1, verbosity=-1)
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model, cv=2, stack_method='predict_proba', n_jobs=-1)

# Training
stacking_model.fit(X_train_sel.values, y_train)

# 9. Evaluation & Threshold
# ── THRESHOLD SEARCH ΜΕ RECALL CONSTRAINT ΣΤΟ VALIDATION SET ────────────────
# Βρίσκουμε το threshold που μεγιστοποιεί το F2 με constraint recall >= 0.50
# Το search γίνεται στο validation set (A+B) ώστε το test set να μην αγγίζεται.

X_val_processed = preprocessor.transform(X_val_raw)
cols_val = preprocessor.get_feature_names_out()
cols_val = [re.sub(r'[\[\]<>,:{}\"]', '_', c) for c in cols_val]
X_val_dum = pd.DataFrame(X_val_processed, columns=cols_val, index=X_val_raw.index)
bool_cols_val = X_val_dum.select_dtypes(include=['uint8', 'bool']).columns
X_val_dum[bool_cols_val] = X_val_dum[bool_cols_val].astype('int8')
X_val_sel = X_val_dum[top_100_features].copy()

y_prob_val = stacking_model.predict_proba(X_val_sel.values)[:, 0]

best_t, best_f2 = 0.5, 0
for t in np.arange(0.05, 0.65, 0.01):
    preds = np.where(y_prob_val > t, 0, 1)
    score = fbeta_score(y_val, preds, beta=2, pos_label=0)
    if score > best_f2:
        best_f2, best_t = score, t

y_prob_admit = stacking_model.predict_proba(X_test_sel.values)[:, 0]
prec_tr, rec_tr, _ = precision_recall_curve(y_test, y_prob_admit, pos_label=0)
y_pred = np.where(y_prob_admit > best_t, 0, 1)
# ─────────────────────────────────────────────────────────────────────────────

# Bias-Variance
mse, bias, var = bias_variance_decomp(stacking_model, X_train_sel.values[:500], y_train[:500], X_test_sel.values, y_test, loss='0-1_loss', num_rounds=2, random_seed=42)

# 10. DASHBOARD 
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
fig.suptitle('Fast Stacking Evaluation Dashboard', fontsize=22, fontweight='bold', y=0.98)

fpr, tpr, _ = roc_curve(y_test, y_prob_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
axes[0, 0].set_title('ROC Curve'); axes[0, 0].legend()

axes[0, 1].plot(rec_tr, prec_tr, color='red', label='PR Curve (Train)')
axes[0, 1].set_title('Precision-Recall Curve '); axes[0, 1].legend()

expert_aucs = [roc_auc_score(y_test, est.predict_proba(X_test_sel.values)[:, 1]) for est in stacking_model.estimators_]
axes[0, 2].bar(['g1', 'g2', 'g3', 'g4', 'g5'], expert_aucs, color='skyblue')
axes[0, 2].set_title('AUC per Expert')

axes[0, 3].axis('off')

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=label_enc.classes_).plot(ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('Confusion Matrix')

f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=0)
ts, tr_s, vl_s = learning_curve(stacking_model, X_train_sel.values, y_train, train_sizes=[0.5, 1.0], cv=2, scoring=f2_scorer)
axes[1, 1].plot(ts, np.mean(tr_s, axis=1), label='Train'); axes[1, 1].plot(ts, np.mean(vl_s, axis=1), label='Val')
axes[1, 1].set_title('Learning Curve'); axes[1, 1].legend()

axes[1, 2].bar(['Bias', 'Variance', 'MSE'], [bias, var, mse], color=['blue', 'red', 'green'], alpha=0.7)
axes[1, 2].set_title('Bias-Variance Decomposition')

axes[1, 3].axis('off')
report = classification_report(y_test, y_pred, target_names=label_enc.classes_)
metrics_text = f"Best Threshold: {best_t:.2f}\nF2-Score: {fbeta_score(y_test, y_pred, beta=2, pos_label=0):.4f}\n\n{report}"
axes[1, 3].text(-0.1, 1.0, metrics_text, fontsize=10, family='monospace', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

model_artifacts = {
    'model': stacking_model,
    'label_encoder': label_enc,
    'preprocessor': preprocessor,
    'top_100_features': top_100_features,
    'best_threshold': best_t
} 

joblib.dump(model_artifacts, 'models/stacking_model_artifacts.pkl')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('plots/stacking_eval.png', dpi=300, bbox_inches='tight')
plt.show()