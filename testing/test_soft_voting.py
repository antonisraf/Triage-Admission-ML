import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, fbeta_score

# Load artifacts saved by the training script
artifacts       = joblib.load('models/soft_voting_artifacts.pkl')
preprocessor    = artifacts['preprocessor']
final_model     = artifacts['model']
label_enc       = artifacts['label_encoder']
top_features    = artifacts['top_features']
final_threshold = artifacts['final_threshold']

# Load & preprocess test set with the SAME pipeline
test_set   = pd.read_csv('data/1stproject-TestSet.csv')
X_test_new = test_set.drop('disposition', axis=1)
y_test_new = label_enc.transform(test_set['disposition'])

X_proc  = preprocessor.transform(X_test_new)
X_ready = X_proc[top_features]

y_prob = final_model.predict_proba(X_ready.values)[:, 0]
y_pred = np.where(y_prob > final_threshold, 0, 1)

print("="*50)
print("  TEST SET PERFORMANCE (SOFT VOTING MODEL)")
print("="*50)
print(classification_report(y_test_new, y_pred, target_names=label_enc.classes_))

f2 = fbeta_score(y_test_new, y_pred, beta=2, pos_label=0)
print(f"F2-Score (Admit Focus): {f2:.4f}")