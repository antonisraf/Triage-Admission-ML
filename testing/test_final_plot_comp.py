import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, fbeta_score
)

# Load the test set
test_set = pd.read_csv('data/1stproject-TestSet.csv')

# Load the artifacts of the Soft Voting model
sv_artifacts    = joblib.load('models/soft_voting_artifacts.pkl')
sv_preprocessor = sv_artifacts['preprocessor']
sv_model        = sv_artifacts['model']
sv_threshold    = sv_artifacts['final_threshold']
sv_top_features = sv_artifacts['top_features']
le_sv           = sv_artifacts['label_encoder']

X_sv_raw = test_set.drop('disposition', axis=1)
y_test   = le_sv.transform(test_set['disposition'])
X_sv     = sv_preprocessor.transform(X_sv_raw)[sv_top_features]

sv_prob = sv_model.predict_proba(X_sv.values)[:, 0]
sv_pred = np.where(sv_prob > sv_threshold, 0, 1)

# Load the functions and the artifacts of the stacking model
def correct_prior(p, train_prior, test_prior):
    odds = p / (1 - p + 1e-9)
    corrected_odds = odds * (test_prior / train_prior) / ((1 - test_prior) / (1 - train_prior))
    return corrected_odds / (1 + corrected_odds)


def engineer_features(df):
    d = df.copy()
    if 'num__esi' in d.columns:
        d['fe_esi_critical']   = (d['num__esi'] <= 2).astype('int8')
        d['fe_esi_borderline'] = (d['num__esi'] == 3).astype('int8')
    if 'num__meds_cardiovascular' in d.columns:
        d['fe_cv_meds_high'] = (d['num__meds_cardiovascular'] > 2).astype('int8')
    if 'num__age' in d.columns and 'num__meds_cardiovascular' in d.columns:
        d['fe_cv_burden_per_decade'] = d['num__meds_cardiovascular'] / (d['num__age'] / 10 + 1e-9)
        d['fe_young_no_cv']          = ((d['num__age'] < 60) & (d['num__meds_cardiovascular'] == 0)).astype('int8')
        d['fe_frailty_proxy']        = (d['num__age'] / 100) * (d.filter(like='meds').sum(axis=1) + 1)
    if all(c in d.columns for c in ['num__esi', 'num__age', 'num__cc_abdominalpain']):
        d['fe_fn_risk']             = ((d['num__esi'] == 3) & (d['num__age'] < 60) & (d['num__cc_abdominalpain'] > 0)).astype('int8')
        d['fe_abdominal_high_risk'] = ((d['num__cc_abdominalpain'] > 0) & (d['num__esi'] <= 3)).astype('int8')
    if 'num__glucose_median' in d.columns and 'num__bun_max' in d.columns:
        d['fe_metabolic_stress']  = ((d['num__glucose_median'] > 140) & (d['num__bun_max'] > 20)).astype('int8')
        d['fe_glucose_bun_ratio'] = d['num__glucose_median'] / (d['num__bun_max'] + 1e-9)
    if 'cat__previousdispo_Admit' in d.columns and 'num__n_admissions' in d.columns:
        d['fe_admission_history'] = d['cat__previousdispo_Admit'] * np.log1p(d['num__n_admissions'])
    if 'num__dbp_min' in d.columns and 'num__spo2_min' in d.columns:
        d['fe_hemodynamic_instability'] = ((d['num__dbp_min'] < 60) | (d['num__spo2_min'] < 94)).astype('int8')
        d['fe_dbp_spo2_combined']       = d['num__dbp_min'] * d['num__spo2_min'] / 100
    if 'num__pregtestur_count' in d.columns and 'num__esi' in d.columns:
        d['fe_gyn_admit_risk'] = ((d['num__pregtestur_count'] > 0) & (d['num__esi'] <= 3)).astype('int8')
    if 'num__cc_abdominalpain' in d.columns and 'num__esi' in d.columns:
        d['fe_abdominal_esi3'] = ((d['num__cc_abdominalpain'] > 0) & (d['num__esi'] == 3)).astype('int8')

    neuro_cols = ['num__cc_unresponsive', 'num__cc_alteredmentalstatus', 'num__cc_lethargy',
                  'num__cc_strokealert', 'num__cc_hallucinations']
    present_neuro = [c for c in neuro_cols if c in d.columns]
    if present_neuro and 'num__esi' in d.columns:
        d['fe_neuro_esi34'] = ((d[present_neuro].sum(axis=1) > 0) & (d['num__esi'].between(3, 4))).astype('int8')

    infect_cols = ['num__cc_fever_75yearsorolder', 'num__cc_respiratorydistress',
                   'num__cc_feverimmunocompromised', 'num__cc_cellulitis', 'num__cc_follow_upcellulitis']
    present_infect = [c for c in infect_cols if c in d.columns]
    if present_infect and 'num__age' in d.columns and 'num__esi' in d.columns:
        d['fe_infectious_risk']    = ((d[present_infect].sum(axis=1) > 0) & (d['num__esi'].between(3, 4))).astype('int8')
        d['fe_elderly_infectious'] = ((d[present_infect].sum(axis=1) > 0) & (d['num__age'] >= 75)).astype('int8')

    gi_alc_cols = ['num__cc_emesis', 'num__cc_epigastricpain',
                   'num__cc_alcoholproblem', 'num__cc_withdrawal_alcohol']
    present_gi = [c for c in gi_alc_cols if c in d.columns]
    if present_gi and 'num__esi' in d.columns:
        d['fe_gi_alc_esi34'] = ((d[present_gi].sum(axis=1) > 0) & (d['num__esi'].between(3, 4))).astype('int8')
    if 'num__age' in d.columns and 'num__meds_cardiovascular' in d.columns:
        d['fe_stealth_elderly'] = ((d['num__age'] >= 65) & (d['num__meds_cardiovascular'] == 0) & (d['num__esi'] == 3)).astype('int8')
    return d


def apply_fn_reduction(prob, X_df, global_t):
    preds = np.where(prob > global_t, 0, 1)

    high_risk_fn = (X_df.get('fe_abdominal_esi3', 0) == 1).values
    preds[high_risk_fn & (prob > 0.02)] = 0

    medium_risk_fn = (
        (X_df.get('fe_fn_risk', 0) == 1) |
        (X_df.get('fe_esi_borderline', 0) == 1)
    ).values
    preds[medium_risk_fn & (prob > 0.03)] = 0

    if 'num__esi' in X_df.columns:
        preds[(X_df['num__esi'] <= 2).values] = 0

    if 'fe_frailty_proxy' in X_df.columns:
        frail_risk = (X_df['fe_frailty_proxy'] > 3.0).values
        preds[frail_risk & (prob > 0.05)] = 0

    return preds


# Stacking artifacts and prediction
st_artifacts        = joblib.load('models/stacking_model_artifacts.pkl')
st_model            = st_artifacts['model']
st_preprocessor     = st_artifacts['preprocessor']
top_100             = st_artifacts['features']['top_100']
final_feature_order = st_artifacts['features']['final_order']
st_threshold        = st_artifacts['thresholds']['best_t']

metadata    = st_artifacts.get('metadata', {})
train_prior = metadata.get('train_prior', 0.30)
test_prior  = metadata.get('test_prior', 0.15)

X_st_raw = test_set.drop('disposition', axis=1).copy()
X_proc   = st_preprocessor.transform(X_st_raw)
cols     = [re.sub(r'[\[\]<>,:{}\"]', '_', c) for c in st_preprocessor.get_feature_names_out()]

X_dum     = pd.DataFrame(X_proc, columns=cols, index=X_st_raw.index)
bool_cols = X_dum.select_dtypes(include=['uint8', 'bool']).columns
X_dum[bool_cols] = X_dum[bool_cols].astype('int8')

X_st = X_dum[top_100].copy()
X_st = engineer_features(X_st)
X_st = X_st[final_feature_order]

st_prob_raw = st_model.predict_proba(X_st.values)[:, 0]
st_prob     = correct_prior(st_prob_raw, train_prior, test_prior)
st_pred     = apply_fn_reduction(st_prob, X_st, st_threshold)

classes = le_sv.classes_

# Dashboard
fig = plt.figure(figsize=(24, 16))
fig.suptitle('Model Comparison: Soft Voting vs Stacking', fontsize=22, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.38, wspace=0.35)

COLORS = {'sv': '#2196F3', 'st': '#FF5722'}

# ROC Curve
ax_roc = fig.add_subplot(gs[0, 0])
for prob, y, label, c in [
    (sv_prob, y_test, 'Soft Voting', COLORS['sv']),
    (st_prob, y_test, 'Stacking',   COLORS['st'])
]:
    fpr, tpr, _ = roc_curve(y, prob, pos_label=0)
    ax_roc.plot(fpr, tpr, lw=2, color=c, label=f'{label} (AUC={auc(fpr, tpr):.3f})')
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
ax_roc.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax_roc.set_xlabel('FPR')
ax_roc.set_ylabel('TPR')
ax_roc.legend(fontsize=9)

# Admit Class Metrics Bar Chart
ax_bar = fig.add_subplot(gs[0, 1])
sv_rep = classification_report(y_test, sv_pred, target_names=classes, output_dict=True)
st_rep = classification_report(y_test, st_pred, target_names=classes, output_dict=True)
sv_f2  = fbeta_score(y_test, sv_pred, beta=1.5, pos_label=0)
st_f2  = fbeta_score(y_test, st_pred, beta=1.5, pos_label=0)

metrics = ['Precision', 'Recall', 'F2-Score']
sv_vals = [sv_rep['Admit']['precision'], sv_rep['Admit']['recall'], sv_f2]
st_vals = [st_rep['Admit']['precision'], st_rep['Admit']['recall'], st_f2]

x = np.arange(len(metrics))
w = 0.32
bars_sv = ax_bar.bar(x - w / 2, sv_vals, w, label='Soft Voting', color=COLORS['sv'], alpha=0.85)
bars_st = ax_bar.bar(x + w / 2, st_vals, w, label='Stacking',   color=COLORS['st'], alpha=0.85)
for bar in list(bars_sv) + list(bars_st):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics)
ax_bar.set_ylim(0, 1.1)
ax_bar.set_title('Admit Class Metrics', fontsize=13, fontweight='bold')
ax_bar.legend(fontsize=9)

# Probability Distributions
for col_idx, (prob, pred, label, thresh) in enumerate([
    (sv_prob, sv_pred, f'Soft Voting  (t={sv_threshold:.2f})', sv_threshold),
    (st_prob, st_pred, f'Stacking  (t={st_threshold:.2f})',    st_threshold)
], start=2):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.hist(prob[y_test == 0], bins=40, alpha=0.6, color='green', label='True Admit')
    ax.hist(prob[y_test == 1], bins=40, alpha=0.6, color='red',   label='True Discharge')
    ax.axvline(thresh, color='black', linestyle='--', lw=1.5, label='Threshold')
    ax.set_title(f'Prob. Distribution\n{label}', fontsize=11, fontweight='bold')
    ax.set_xlabel('P(Admit)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)

# Confusion Matrices
for col_idx, (pred, label) in enumerate([
    (sv_pred, 'Soft Voting'),
    (st_pred, 'Stacking')
], start=1):
    ax = fig.add_subplot(gs[1, col_idx])
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred),
        display_labels=classes
    ).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix\n{label}', fontsize=11, fontweight='bold')

# Summary Table
ax_txt = fig.add_subplot(gs[1, 0])
ax_txt.axis('off')
summary = (
    f"{'Metric':<18} {'SV':>6} {'ST':>6}\n"
    f"{'-'*32}\n"
    f"{'Admit Precision':<18} {sv_rep['Admit']['precision']:>6.3f} {st_rep['Admit']['precision']:>6.3f}\n"
    f"{'Admit Recall':<18} {sv_rep['Admit']['recall']:>6.3f} {st_rep['Admit']['recall']:>6.3f}\n"
    f"{'Admit F1':<18} {sv_rep['Admit']['f1-score']:>6.3f} {st_rep['Admit']['f1-score']:>6.3f}\n"
    f"{'F2-Score':<18} {sv_f2:>6.3f} {st_f2:>6.3f}\n"
    f"{'Accuracy':<18} {sv_rep['accuracy']:>6.3f} {st_rep['accuracy']:>6.3f}\n"
    f"{'Threshold':<18} {sv_threshold:>6.3f} {st_threshold:>6.3f}\n"
)
ax_txt.text(0.05, 0.88, summary, fontsize=9, family='monospace', va='top',
            transform=ax_txt.transAxes, clip_on=True,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.4))
ax_txt.set_title('Summary', fontsize=13, fontweight='bold')

# Overall Accuracy Bar
ax_acc = fig.add_subplot(gs[1, 3])
acc_vals = [sv_rep['accuracy'], st_rep['accuracy']]
bars = ax_acc.bar(['Soft Voting', 'Stacking'], acc_vals,
                  color=[COLORS['sv'], COLORS['st']], alpha=0.85, width=0.4)
for bar, v in zip(bars, acc_vals):
    ax_acc.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax_acc.set_ylim(0, 1.1)
ax_acc.set_title('Overall Accuracy', fontsize=13, fontweight='bold')

plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()