import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna 
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
import lightgbm as lgb
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score 
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, fbeta_score, 
                             classification_report, confusion_matrix, roc_auc_score, make_scorer,
                             ConfusionMatrixDisplay)
from catboost import CatBoostClassifier
from mlxtend.evaluate import bias_variance_decomp
from sklearn.impute import SimpleImputer

# I reduced the dataset to 40% of the original size and saved it as 'subset.csv'

#df=pd.read_csv('1stproject.csv')
#df_subset= df.sample(frac=0.4,random_state=1)
#df_subset.to_csv('subset.csv',index=False) 

df = pd.read_csv('data/subset.csv')
X = df.drop('disposition', axis=1) 
y = df['disposition']

X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)


# Then i transformed my target value into 0/1 where 0 is "Admit" and 1 is "Discharge"

label_enc = LabelEncoder()
y_train = label_enc.fit_transform(y_train_raw)
y_val = label_enc.transform(y_val_raw)
y_test = label_enc.transform(y_test_raw)

# Setting up the preprocessing pipelines for numeric and categorical features
numeric_cols = X_train_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train_raw.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
], verbose_feature_names_out=False)

# Change the output from array to pandas dataframe
preprocessor.set_output(transform="pandas")


preprocessor.fit(X_train_raw)

# Transform the datasets
X_train_processed_arr = preprocessor.transform(X_train_raw)
X_val_processed_arr = preprocessor.transform(X_val_raw)
X_test_processed_arr = preprocessor.transform(X_test_raw)

# Get feature names after transformation
cat_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_names)

# Convert the processed arrays back to DataFrames with appropriate column names
X_train_processed = pd.DataFrame(X_train_processed_arr, columns=all_feature_names, index=X_train_raw.index)
X_val_processed = pd.DataFrame(X_val_processed_arr, columns=all_feature_names, index=X_val_raw.index)
X_test_processed = pd.DataFrame(X_test_processed_arr, columns=all_feature_names, index=X_test_raw.index)

# Feature Selection: Apply Variance Threshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_train_processed)

# Get the columns that passed the variance threshold
selected_cols = X_train_processed.columns[selector.get_support()]
X_train_sel = X_train_processed[selected_cols]
X_val_sel = X_val_processed[selected_cols]
X_test_sel = X_test_processed[selected_cols]

# Performed a Random forest in order to get the top 100 features
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_selector.fit(X_train_sel, y_train)
importances = pd.Series(rf_selector.feature_importances_, index=X_train_sel.columns)
top_features = importances.nlargest(100).index.tolist()

# Split the features into 4 groups based on keywords
vitals_top = [c for c in top_features if any(w in c.lower() for w in ['vital', 'sbp', 'dbp', 'pulse', 'temp', 'o2', 'hr', 'rr'])]
meds_top = [c for c in top_features if c.startswith('meds_') and c not in vitals_top]
labs_top = [c for c in top_features if any(w in c.lower() for w in ['median', 'min', 'max', 'last']) and c not in vitals_top and c not in meds_top]
history_top = [c for c in top_features if c not in vitals_top + meds_top + labs_top]

X_train = X_train_sel[top_features]
X_val = X_val_sel[top_features]
X_test = X_test_sel[top_features]

def get_indices(df, col_list):
    return [df.columns.get_loc(c) for c in col_list]

vitals_idx = get_indices(X_train, vitals_top)
meds_idx = get_indices(X_train, meds_top)
labs_idx = get_indices(X_train, labs_top)
history_idx = get_indices(X_train, history_top)

def custom_admit_scorer(y_true, y_pred):
    prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    if prec < 0.60:
        return fbeta_score(y_true, y_pred, beta=2, pos_label=0) * ((prec / 0.60) ** 2)
    return fbeta_score(y_true, y_pred, beta=2, pos_label=0)

f2_scorer = make_scorer(custom_admit_scorer)

# Optuna objective function
def objective(trial):
    lgb_lr = trial.suggest_float('lgb_lr', 0.01, 0.1,log=True)
    lgb_reg = trial.suggest_float('lgb_reg', 1.0, 20.0,log=True)
    log_c = trial.suggest_float('log_c', 0.001, 100.0,log=True)
    cat_depth = trial.suggest_int('cat_depth', 4, 8) 
    cat_lr = trial.suggest_float('cat_lr', 0.01, 0.1,log=True)
    
    w_vit = trial.suggest_int('w_vit', 1, 5)
    w_med = trial.suggest_int('w_med', 1, 5)
    w_lab = trial.suggest_int('w_lab', 1, 5)
    w_his = trial.suggest_int('w_his', 1, 5)

    base_learners_trial = [
        ('vitals_expert', Pipeline([
            ('sel', ColumnTransformer([('keep', 'passthrough', vitals_idx)], remainder='drop')),
            ('clf', lgb.LGBMClassifier(n_estimators=200, learning_rate=lgb_lr, reg_lambda=lgb_reg, random_state=42, n_jobs=-1, verbosity=-1)) 
        ])),  
        ('meds_expert', Pipeline([
            ('sel', ColumnTransformer([('keep', 'passthrough', meds_idx)], remainder='drop')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=500, C=log_c, random_state=42, n_jobs=-1))
        ])), 
        ('labs_expert', Pipeline([
            ('sel', ColumnTransformer([('keep', 'passthrough', labs_idx)], remainder='drop')),
            ('clf', CatBoostClassifier(iterations=250, verbose=0, learning_rate=cat_lr, depth=cat_depth, thread_count=-1))
        ])), 
        ('history_expert', Pipeline([
            ('sel', ColumnTransformer([('keep', 'passthrough', history_idx)], remainder='drop')),
            ('clf', CatBoostClassifier(iterations=250, verbose=0, learning_rate=cat_lr, depth=cat_depth, thread_count=-1))
        ]))
    ]

    Voting_model = VotingClassifier(estimators=base_learners_trial, voting='soft', weights=[w_vit, w_med, w_lab, w_his])
    score = cross_val_score(Voting_model, X_train.values, y_train, cv=3, scoring='average_precision', n_jobs=-1)
    return score.mean()

# Bayesian Optimization
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=15)

# Final model with best parameters
bp = study.best_params
base_learners_final = [
    ('vitals_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', vitals_idx)], remainder='drop')),
        ('clf', lgb.LGBMClassifier(n_estimators=200, learning_rate=bp['lgb_lr'], reg_lambda=bp['lgb_reg'], class_weight='balanced',random_state=42, n_jobs=-1, verbosity=-1)) 
    ])),  
    ('meds_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', meds_idx)], remainder='drop')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, C=bp['log_c'],class_weight='balanced', random_state=42, n_jobs=-1))
    ])), 
    ('labs_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', labs_idx)], remainder='drop')),
        ('clf', CatBoostClassifier(iterations=250, verbose=0, learning_rate=bp['cat_lr'], depth=bp['cat_depth'],thread_count=-1,random_seed=42))
    ])), 
    ('history_expert', Pipeline([
        ('sel', ColumnTransformer([('keep', 'passthrough', history_idx)], remainder='drop')),
        ('clf', CatBoostClassifier(iterations=250, verbose=0, learning_rate=bp['cat_lr'], depth=bp['cat_depth'], thread_count=-1,random_seed=42))
    ]))
]

final_model = VotingClassifier(estimators=base_learners_final, voting='soft', weights=[bp['w_vit'], bp['w_med'], bp['w_lab'], bp['w_his']])
final_model.fit(X_train.values, y_train)


y_prob_train_admit = final_model.predict_proba(X_train.values)[:, 0]
y_prob_val_admit = final_model.predict_proba(X_val.values)[:, 0]
y_prob_test_admit = final_model.predict_proba(X_test.values)[:, 0]

thresholds_to_try = np.arange(0.20, 0.65, 0.01)
best_thresh, best_f2 = 0.5, 0

for t in thresholds_to_try:
    preds = np.where(y_prob_val_admit > t, 0, 1)
    prec = precision_score(y_val, preds, pos_label=0, zero_division=0)
    if prec < 0.60:
        continue
    score = fbeta_score(y_val, preds, beta=2, pos_label=0)
    if score > best_f2:
        best_f2, best_thresh = score, t

final_threshold = best_thresh

# Final predictions using the adjusted threshold
y_pred_custom = np.where(y_prob_test_admit > final_threshold, 0, 1)


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

# Dashboard Setup 
fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle('Soft-Voting Evaluation Dashboard', fontsize=22, fontweight='bold', y=0.98)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_test_admit, pos_label=0)
axes[0, 0].plot(fpr, tpr, color='blue', label=f'AUC = {auc(fpr, tpr):.2f}')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()

# PR Curve
precision_tr, recall_tr, thresholds_tr = precision_recall_curve(y_val, y_prob_val_admit, pos_label=0)
axes[0, 1].plot(recall_tr, precision_tr, color='red', label='PR Curve (Val)')
axes[0, 1].set_title('Precision-Recall Curve')
axes[0, 1].legend()

# AUC per Expert
expert_names = [name.replace('_expert', '') for name, _ in final_model.estimators]
expert_aucs = [roc_auc_score(y_test, final_model.estimators_[i].predict_proba(X_test.values)[:, 0]) for i in range(len(expert_names))]
axes[0, 2].bar(expert_names, expert_aucs, color='skyblue')
axes[0, 2].set_title('AUC per Expert')

# Confusion Matrix
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_).plot(ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('Confusion Matrix')

# Learning Curve
axes[1, 1].plot(train_sizes, train_mean, label='Train')
axes[1, 1].plot(train_sizes, val_mean, label='Val')
axes[1, 1].set_title('Learning Curve')
axes[1, 1].legend()

# Bias Varinace Bar Chart
sample_size = min(5000, len(X_train))
indices = np.random.choice(len(X_train), sample_size, replace=False)
mse, bias, var = bias_variance_decomp(
    final_model, 
    X_train.values[indices], y_train[indices], 
    X_test.values, y_test, 
    loss='0-1_loss', num_rounds=5, random_seed=42
)
axes[1, 2].bar(['Bias', 'Variance', 'MSE'], [bias, var, mse], color=['blue', 'red', 'green'], alpha=0.7)
axes[1, 2].set_title('Bias-Variance Decomposition')
for i, v in enumerate([bias, var, mse]): 
    axes[1, 2].text(i, v + 0.005, f'{v:.3f}', ha='center')

model_artifacts = {
    'preprocessor': preprocessor,
    'model': final_model,
    'label_encoder': label_enc,
    'top_features': top_features,
    'final_threshold': final_threshold
}

joblib.dump(model_artifacts, 'models/soft_voting_artifacts.pkl')

# Metrics Text 
plt.figtext(0.85, 0.25, f"Applied Threshold: {final_threshold:.2f}\nF2-Score: {f2_admit:.4f}\n\n{report}", 
            fontsize=10, family='monospace', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])

plt.savefig('plots/soft_voting_eval.png', dpi=300, bbox_inches='tight')
plt.show()