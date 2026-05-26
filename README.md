# Triage Admission Machine Learning Project

> Predicting patient triage admission using ensemble machine learning, featuring a performance comparison between Stacking and Soft Voting classifiers.

---

# Credits & Acknowledgements

This project was built upon and utilizes the dataset provided by the following research study:

**Original Paper:**  
Hong WS, Haimovich AD, Taylor RA (2018).  
*Predicting hospital admission at emergency department triage using machine learning*.  
PLOS ONE 13(7): e0201016.

**DOI:**  
https://doi.org/10.1371/journal.pone.0201016

## Data Availability

The de-identified, processed dataset of patient visits, along with the original scripts used by the authors for processing and analysis, are publicly available at:

- **GitHub Repository:**  
  https://github.com/yaleemmlc/admissionprediction

- **Zenodo:**  
  https://doi.org/10.5281/zenodo.1308993

## License

The original study and its associated materials are distributed under the terms of the **Creative Commons Attribution License**, which permits unrestricted use, distribution, and reproduction in any medium, provided the original authors and source are credited.

---

# Dataset Preprocessing

## Create the `subset.csv`

Unlike the original study, the primary dataset used for training and validation in this project was pre-filtered to include only patient records from **Departments A and B**. Furthermore, due to hardware constraints, a 40% random sample of this filtered data was extracted (`subset.csv`) to serve as the main training set.

For the final out-of-domain evaluation, a separate dataset containing records exclusively from **Department C** was used as the test set (`1stproject-TestSet.csv`).

```python
df = pd.read_csv('1stproject.csv')

df_subset = df.sample(frac=0.4, random_state=1)

df_subset.to_csv('subset.csv', index=False)
```

This deliberate train/test split across departments simulates a realistic **domain shift** scenario: the models are trained on data from one hospital environment and evaluated on a completely different one. This is a more honest evaluation than an in-distribution split and directly tests generalization.

---

# Modeling Approach

The strategy was conducted in two main phases to build and evaluate two distinct ensemble models.

**Phase 1 — Soft Voting Classifier (Baseline Ensemble):**  
A domain-expert architecture where specialized base learners are trained on semantically grouped feature subsets and their predictions are combined via weighted average. This approach provides interpretability and control over which clinical domain each model is responsible for.

**Phase 2 — Stacking Classifier (Advanced Ensemble):**  
A meta-learning architecture that replaces static voting weights with a learned combiner (meta-learner). The meta-learner is trained on the cross-validated probability outputs of the base models, allowing it to discover non-obvious synergies between them.

The two architectures are then evaluated side-by-side on the out-of-domain test set to measure which one generalizes better under distribution shift.

---

# 1. Soft Voting Classifier

For the initial modeling phase, a **Soft Voting Ensemble** was implemented utilizing a "Domain Expert" architecture. Instead of feeding all features indiscriminately into a single model, the data was semantically grouped to train specialized base learners, each responsible for a distinct clinical domain.

## Data Preprocessing & Feature Selection

The raw data was split into **Train (70%) / Validation (15%) / Test (15%)** sets using a stratified approach to preserve the class ratio across splits.

- **Numeric features:** Missing values imputed with the **median**.
- **Categorical features:** Missing values imputed with the **most frequent value**, followed by **One-Hot Encoding**.
- **Dimensionality reduction:** A `VarianceThreshold` was applied first to remove zero-variance (constant) predictors. A `RandomForestClassifier` was then used to rank features by importance, and the **Top 100** were retained.

## The "Domain Expert" Architecture

The 100 selected features were parsed and separated into four distinct semantic groups using keyword matching. Each group was then assigned to a specific algorithmic expert best suited for that type of data:

| Expert | Algorithm | Feature Domain |
|---|---|---|
| Vitals Expert | `LGBMClassifier` | Physiological measurements (pulse, BP, O2, etc.) |
| Medications Expert | `LogisticRegression` | Patient medication history |
| Labs Expert | `CatBoostClassifier` | Laboratory test results (min, max, median values) |
| History Expert | `CatBoostClassifier` | Medical history & demographics |

> **Note on Medications Expert:** The medication feature subset was additionally scaled using `StandardScaler` before being passed to the Logistic Regression model, since linear models are sensitive to feature magnitude.

## Hyperparameter Tuning & Optimization

**Optuna** was used for Bayesian Optimization to simultaneously tune:

- The internal hyperparameters of each base learner (learning rates, tree depths, regularization terms).
- The voting weights (`w_vit`, `w_med`, `w_lab`, `w_his`) assigned to each expert in the final ensemble.

The optimization objective was a **custom scoring function based on the F2-Score**. The F2-Score weights Recall twice as heavily as Precision, directly penalizing False Negatives (missed admissions) — a critical consideration in clinical triage where failing to admit a high-risk patient is far more dangerous than an unnecessary admission.

To prevent the model from collapsing into a high-recall/near-zero-precision classifier, the scoring function imposed a **hard constraint**: if Precision dropped below `0.60` on the validation set, the trial was penalized regardless of its Recall score.

## Final Inference

The final tuned Voting Classifier produces the **weighted average probability** across all four expert models. The decision threshold was fine-tuned on the validation set to maximize the F2-Score before making final Admit/Discharge predictions on the test set.

All necessary components — trained models, preprocessor, feature group lists, and the optimal threshold — were serialized as `soft_voting_artifacts.pkl` for reproducible inference.

[![Soft Voting Training Results](plots/soft_voting_eval.png)](plots/soft_voting_eval.png)

## Training Evaluation

**ROC Curve — AUC = 0.91**, indicating excellent discriminative ability between the Admit and Discharge classes.

**Precision-Recall Curve** remains consistently high up to approximately Recall = 0.70, after which it drops sharply. This confirms that the choice of a low threshold (`0.34`) was a deliberate clinical priority: maximizing the recovery of Admit cases at the cost of some precision.

**AUC per Expert** reveals the relative contribution of each domain:

| Expert | AUC |
|---|---|
| Labs | ~0.33 |
| Vitals | ~0.23 |
| Medications | ~0.21 |
| History | ~0.14 |

**Confusion Matrix** (n = 14,015, threshold = 0.34): the model recovers **89% of true Admit cases**, at the cost of 2,748 False Positives.

**Learning Curve**: training score starts high (~0.81) and stabilizes around 0.71, while validation rises gradually to ~0.69–0.70. The small and shrinking gap between the two curves confirms good generalization with no significant overfitting.

**Bias-Variance Decomposition** (5,000-sample subset, num_rounds=5, 0-1 loss):
- **Variance = 0.027** — extremely low, confirming model stability.
- **Bias = 0.163** — the dominant source of error.
- **MSE ≈ 0.165** — essentially equal to Bias², confirming variance is not a concern.

---

# 2. Stacking Classifier

To move beyond the static weights of the Soft Voting model, a **Stacking Classifier** was designed. This architecture replaces manual weight assignment with a trained meta-learner that learns how to best combine the outputs of the base models — including learning when to trust or discount each one.

## Feature Selection & Clinical Feature Engineering

Following the standard preprocessing pipeline (imputation and One-Hot Encoding), a `LGBMClassifier` was trained on a subset of the training data to rank features by **Information Gain**, and the **Top 100** were selected.

To specifically combat False Negatives, **14 custom clinical features (`fe_*`)** were engineered on top of the selected features. These features capture interaction effects targeting high-risk patient profiles that standard features may under-represent — for example, patients with borderline ESI scores combined with severe abdominal pain and subtle metabolic stress indicators. Feature engineering of this type encodes domain knowledge directly into the model's input space.

## Feature Subspacing (Model Diversity)

A key requirement for effective stacking is that the base learners must be **diverse** — they should make different errors, so that the meta-learner has something to learn from their combination. If all base models are trained on identical feature sets, they tend to correlate strongly and offer diminishing returns when combined.

To enforce diversity, **Feature Subspacing** was used: 5 distinct feature sets were created from the training data. Each set contained:

1. The **top 5 anchor features** (shared across all sets for stability).
2. A **randomized subset** of the remaining 95 features (unique per set).
3. All **14 engineered clinical features** (shared across all sets).

## Ensemble Architecture & Training

Each of the 5 feature sets was fed into a distinct base learner:

| Base Learner | Notes |
|---|---|
| `LGBMClassifier` | Fast gradient boosting, handles sparse features well |
| `RandomForestClassifier` | Bagging-based diversity, robust to noise |
| `CatBoostClassifier` (init A) | Handles categorical features natively |
| `CatBoostClassifier` (init B) | Alternative initialization for additional diversity |
| `XGBClassifier` | Regularized gradient boosting |

The **probability outputs** (not class predictions) of these base learners were generated via **cross-validation during the stacking fit** — this prevents data leakage, since the meta-learner never sees predictions made on the same data each base model was trained on.

These out-of-fold probabilities were then passed to a **Logistic Regression Meta-Learner**, which learned the optimal linear combination of the base model outputs.

## Hyperparameter Tuning

**Optuna** was again used to optimize the learning rates across all gradient boosting base models and the meta-learner's regularization parameter simultaneously.

The optimization target was a **custom F1.5-Score**, which weights Recall more than Precision (but less aggressively than F2), selected to push the model toward minimizing missed admissions while maintaining stronger precision than the Soft Voting approach.

## Final Training Artifacts

After finding the optimal hyperparameters, the final Stacking Classifier was retrained on the **entire preprocessed training set**. The complete pipeline — including the trained stacking model, base learners, preprocessor, and feature lists — was serialized as `stacking_model_artifacts.pkl` for inference.

![Stacking Training Results](plots/stacking_eval.png)

## Training Evaluation

**ROC Curve — AUC = 0.92**, slightly higher than Soft Voting (0.91), confirming equally excellent discriminative ability between Admit and Discharge.

**Precision-Recall Curve** remains very high up to approximately Recall = 0.80 before dropping sharply. Since this curve is evaluated on the training set, it reflects how well the model has learned the training patterns rather than its generalization ability.

**Confusion Matrix** (n = 14,015, threshold = 0.40): the model correctly classifies 9,084 Discharge and 2,981 Admit cases, with 484 False Positives and 1,466 False Negatives. Compared to Soft Voting, Stacking achieves better Admit Precision (0.86 vs 0.59) but lower Recall (0.67 vs 0.89) — a direct consequence of the higher threshold (0.40 vs 0.34) chosen based on the F1.5-Score objective rather than F2.

**Learning Curve**: training score starts very high (~0.84) and continues declining without fully stabilizing (~0.78), while validation rises slowly to ~0.74. The gap between train and validation remains noticeably larger than in Soft Voting, indicating more pronounced overfitting.

**Bias-Variance Decomposition** (random training subset, num_rounds=5, 0-1 loss):
- **Variance ≈ 0.040** — slightly higher than Soft Voting (0.027), consistent with the larger Learning Curve gap.
- **Bias ≈ 0.175** — remains the dominant source of error.
- **MSE ≈ 0.170** — reflects the sum of both components.

Overall: **F1.5-Score = 0.7192**, accuracy = 0.86, weighted F1 = 0.86.

# Workflows

## 1. Soft Voting Classifier

![Soft Voting Workflow](model_logic_diagrams/Soft_Voting_Workflow.png)

## 2. Stacking Classifier (Advanced Ensemble)

![Stacking Workflow](model_logic_diagrams/Stacking_Workflow.png)

---

# 3. Final Evaluation: Soft Voting vs. Stacking

To conclusively determine the best approach, a unified inference script evaluates both the **Soft Voting Classifier** and the **Stacking Classifier** side-by-side on the completely unseen test set (`1stproject-TestSet.csv`, representing Department C).

## The Unified Inference Pipeline

The evaluation process runs both models through their respective pipelines:

1. Artifact loading.
2. Standard preprocessing.
3. Model-specific transformations and thresholding.
4. Final Admit/Discharge prediction generation.

## Specialized Logic for the Stacking Model

Because the Stacking model was designed to handle complex clinical realities, two additional steps were applied **exclusively to its probability outputs** before final classification.

### Prior Correction

The raw probabilities were mathematically recalibrated to adjust the model's perspective from the artificially balanced training environment (`30%` admit rate) to the real-world expected prevalence (`15%` admit rate).

### 4-Tier False Negative Reduction

A custom **clinical override system** dynamically lowered the admission threshold for specific high-risk patient profiles, acting as a safety net against dangerous False Negatives. Patients falling into predefined high-risk tiers (based on combinations of ESI score, vital signs, and engineered features) were admitted at lower probability cutoffs than the global threshold.

## Visual Dashboard & Metrics

The script generates a comprehensive comparison dashboard (`plots/model_comparison.png`) containing:

- ROC Curves (AUC)
- Admit-class precision, recall, and F-scores
- Probability distributions
- Confusion matrices

![Model Comparison Dashboard](plots/model_comparison.png)

---

# Final Model Comparison: Soft Voting vs. Stacking (Test Set)

## Class Distribution Shift

Before interpreting the results, it is important to note the distribution difference between environments:

| Environment | Admissions | Discharges |
|---|---|---|
| Training (Depts A & B) | ~30% | ~70% |
| Test Set (Dept C) | ~15% | ~85% |

This significant **distribution shift** is the primary source of performance degradation on the test set. Both models were trained in a substantially more balanced environment than they were evaluated on — making the out-of-domain evaluation a realistic but challenging benchmark.

## ROC Analysis & Discriminative Power

| Model | AUC |
|---|---|
| Soft Voting | **0.867** |
| Stacking | 0.824 |

The Soft Voting model retains higher global discriminative ability on the test set, meaning it separates admissions from discharges more cleanly across all possible thresholds. The Stacking model's lower AUC reflects the impact of distribution shift on its probability estimates, which are partially corrected by Prior Correction but not fully recovered.

## Precision-Recall Trade-off

| Metric | Soft Voting | Stacking |
|---|---|---|
| Recall (Safety) | **0.82** | 0.77 |
| Precision (Efficiency) | 0.37 | **0.41** |
| F2-Score | ~0.60 | ~0.60 |
| F1-Score | 0.508 | **0.538** |

- **Recall:** Soft Voting catches more true admissions, minimizing dangerous misses.
- **Precision:** Stacking produces fewer false alarms per actual admission predicted.
- **F2 / F1:** Both models land at comparable F2 performance, but Stacking achieves better overall F1 — indicating a more balanced operating point.

## Overall Conclusion

The **Soft Voting** model demonstrates a more aggressive prediction strategy — it casts a wider net, maximizing safety at the cost of a higher false positive rate. This behavior is by design: the F2-Score optimization and conservative threshold push it to err on the side of caution.

The **Stacking Classifier** exhibits a more balanced and conservative behavior, maintaining strong recall while substantially reducing unnecessary admission predictions. Its use of Prior Correction and clinical override tiers allows it to adapt more gracefully to the real-world class distribution of Department C.

Overall, the results indicate that although both models remain robust under distribution shift, the **Stacking approach generalizes more effectively** to Department C by offering a better trade-off between patient safety and operational efficiency.

---

# Limitations

- **Single-year window:** No seasonal validation. Winter patterns such as respiratory illness spikes may not generalize to other periods.
- **Disposition as ground truth:** The model learns to predict what the hospital historically decided, not necessarily what the patient clinically needed. Systemic biases in admission decisions are absorbed by the model.

---

# Installation & Usage

## 1. Clone the Repository

```bash
git clone https://github.com/antonisraf/Triage-Admission-ML.git
cd Triage-Admission-ML
```

## 2. Create a Virtual Environment (Recommended)

This project was developed using **Python 3.12.6**.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

All required libraries are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 4. Create the Models Directory

```bash
mkdir models
```