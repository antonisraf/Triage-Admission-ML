import pandas as pd
import numpy as np
import re
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# 1. ΦΟΡΤΩΣΗ ARTIFACTS
# ---------------------------------------------------------
artifacts           = joblib.load('models/stacking_model_artifacts.pkl')
model               = artifacts['model']
label_enc           = artifacts['label_encoder']
preprocessor        = artifacts['preprocessor']

top_100             = artifacts['features']['top_100']
engineered_features = artifacts['features']['engineered']
final_feature_order = artifacts['features']['final_order']
best_t              = artifacts['thresholds']['best_t']

metadata            = artifacts.get('metadata', {})
train_prior         = metadata.get('train_prior', 0.30)
test_prior          = metadata.get('test_prior', 0.15)

# 2. ΣΥΝΑΡΤΗΣΕΙΣ & ΠΡΟΕΤΟΙΜΑΣΙΑ
# ---------------------------------------------------------
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
        d['fe_fn_risk'] = ((d['num__esi'] == 3) & (d['num__age'] < 60) & (d['num__cc_abdominalpain'] > 0)).astype('int8')
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
    
    neuro_cols = ['num__cc_unresponsive', 'num__cc_alteredmentalstatus', 'num__cc_lethargy', 'num__cc_strokealert', 'num__cc_hallucinations']
    present_neuro = [c for c in neuro_cols if c in d.columns]
    if present_neuro and 'num__esi' in d.columns:
        d['fe_neuro_esi34'] = ((d[present_neuro].sum(axis=1) > 0) & (d['num__esi'].between(3, 4))).astype('int8')

    infect_cols = ['num__cc_fever_75yearsorolder', 'num__cc_respiratorydistress', 'num__cc_feverimmunocompromised', 'num__cc_cellulitis', 'num__cc_follow_upcellulitis']
    present_infect = [c for c in infect_cols if c in d.columns]
    if present_infect and 'num__age' in d.columns and 'num__esi' in d.columns:
        d['fe_infectious_risk'] = ((d[present_infect].sum(axis=1) > 0) & (d['num__esi'].between(3, 4))).astype('int8')
        d['fe_elderly_infectious'] = ((d[present_infect].sum(axis=1) > 0) & (d['num__age'] >= 75)).astype('int8')

    gi_alc_cols = ['num__cc_emesis', 'num__cc_epigastricpain', 'num__cc_alcoholproblem', 'num__cc_withdrawal_alcohol']
    present_gi = [c for c in gi_alc_cols if c in d.columns]
    if present_gi and 'num__esi' in d.columns:
        d['fe_gi_alc_esi34'] = ((d[present_gi].sum(axis=1) > 0) & (d['num__esi'].between(3, 4))).astype('int8')
    if 'num__age' in d.columns and 'num__meds_cardiovascular' in d.columns:
        d['fe_stealth_elderly'] = ((d['num__age'] >= 65) & (d['num__meds_cardiovascular'] == 0) & (d['num__esi'] == 3)).astype('int8')
    return d

# 3. LOAD TEST SET & PREPROCESS
# ---------------------------------------------------------
test_set = pd.read_csv('data/1stproject-TestSet.csv')
X = test_set.drop('disposition', axis=1).copy()
y = label_enc.transform(test_set['disposition'])

X_proc = preprocessor.transform(X)
cols = [re.sub(r'[\[\]<>,:{}\"]', '_', c) for c in preprocessor.get_feature_names_out()]
X_dum = pd.DataFrame(X_proc, columns=cols, index=X.index)
bool_cols = X_dum.select_dtypes(include=['uint8', 'bool']).columns
X_dum[bool_cols] = X_dum[bool_cols].astype('int8')

X_sel = X_dum[top_100].copy()
X_sel = engineer_features(X_sel)
X_sel = X_sel[final_feature_order]

y_prob_raw = model.predict_proba(X_sel.values)[:, 0]
y_prob     = correct_prior(y_prob_raw, train_prior, test_prior)

def apply_fn_reduction(prob, X_df, global_t):
    # Αρχική πρόβλεψη με το βέλτιστο global threshold
    preds = np.where(prob > global_t, 0, 1)

    # Tier 1: Πολύ επιθετικό threshold (0.02) για την ομάδα "υψηλού κινδύνου FN"
    # Εδώ στοχεύουμε τον συνδυασμό Κοιλιακού Πόνου + ESI 3 (fe_abdominal_esi3)
    high_risk_fn = (X_df.get('fe_abdominal_esi3', 0) == 1).values
    preds[high_risk_fn & (prob > 0.02)] = 0

    # Tier 2: Επιθετικό threshold (0.03) για γενικές FN-prone ομάδες
    # Εδώ στοχεύουμε το fe_fn_risk και το fe_esi_borderline
    medium_risk_fn = (
        (X_df.get('fe_fn_risk', 0) == 1) | 
        (X_df.get('fe_esi_borderline', 0) == 1)
    ).values
    preds[medium_risk_fn & (prob > 0.03)] = 0

    # Tier 3: Hard Rules (Ακαριαίο Admit)
    # Ασθενείς με ESI 1 ή 2 θεωρούνται αυτόματα Admit ανεξαρτήτως πιθανότητας
    if 'num__esi' in X_df.columns:
        preds[(X_df['num__esi'] <= 2).values] = 0

    # Tier 4: Clinical Override (Προστασία ηλικιωμένων με επιβαρυμένη εικόνα)
    # Αν το fe_frailty_proxy είναι πολύ υψηλό, το Discharge είναι ριψοκίνδυνο
    if 'fe_frailty_proxy' in X_df.columns:
        # Χρησιμοποιούμε μια τιμή αποκοπής βάσει του Profile Table (π.χ. > 3.0)
        frail_risk = (X_df['fe_frailty_proxy'] > 3.0).values
        preds[frail_risk & (prob > 0.05)] = 0

    return preds

# Εφαρμογή της νέας λογικής
y_pred = apply_fn_reduction(y_prob, X_sel, best_t)

# 5. EVALUATION & CONFUSION MATRIX
# ---------------------------------------------------------
print("="*60)
print(f" PERFORMANCE REPORT | Global Threshold: {best_t:.4f}")
print("="*60)
print(classification_report(y, y_pred, target_names=label_enc.classes_))

cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, index=[f"True {c}" for c in label_enc.classes_], 
                         columns=[f"Pred {c}" for c in label_enc.classes_])

print("\nCONFUSION MATRIX:")
print("-" * 30)
print(cm_df)

tp = cm[0, 0]
tn = cm[1, 1]
fp = cm[1, 0]
fn = cm[0, 1]

print(f"\n[Summary Counts] TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

# 6. AVERAGE PROFILES PER GROUP
# ---------------------------------------------------------
analysis_df = test_set.copy()
analysis_df['True_Label'] = label_enc.inverse_transform(y)
analysis_df['Pred_Label'] = label_enc.inverse_transform(y_pred)

groups = {
    'TP': (analysis_df['True_Label'] == 'Admit')     & (analysis_df['Pred_Label'] == 'Admit'),
    'TN': (analysis_df['True_Label'] == 'Discharge') & (analysis_df['Pred_Label'] == 'Discharge'),
    'FP': (analysis_df['True_Label'] == 'Discharge') & (analysis_df['Pred_Label'] == 'Admit'),
    'FN': (analysis_df['True_Label'] == 'Admit')     & (analysis_df['Pred_Label'] == 'Discharge')
}

all_features = final_feature_order
profile_num = X_sel[all_features].select_dtypes(include=['number']).columns.tolist()

print("\n" + "="*60)
print(" AVERAGE NUMERIC PROFILE PER GROUP")
print("="*60)
num_profile = pd.DataFrame({name: X_sel.loc[mask, profile_num].mean() for name, mask in groups.items()})
num_profile['FP_vs_TN'] = num_profile['FP'] - num_profile['TN']
num_profile['FN_vs_TP'] = num_profile['FN'] - num_profile['TP']
print(num_profile.round(3).to_string())