# Triage Admission Machine Learning Project

> [About:Predicting patient triage admission using ensemble machine learning, featuring a performance comparison between Stacking and Soft Voting classifiers.]

## Credits & Acknowledgements

This project was built upon and utilizes the dataset provided by the following research study:

* **Original Paper:** Hong WS, Haimovich AD, Taylor RA (2018). *Predicting hospital admission at emergency department triage using machine learning*. PLoS ONE 13(7): e0201016. 
* **DOI:** [https://doi.org/10.1371/journal.pone.0201016](https://doi.org/10.1371/journal.pone.0201016)

**Data Availability:**
The de-identified, processed dataset of patient visits, along with the original scripts used by the authors for processing and analysis, are publicly available at:
* **GitHub Repository:** [yaleemmlc/admissionprediction](https://github.com/yaleemmlc/admissionprediction)
* **Zenodo:** [10.5281/zenodo.1308993](https://doi.org/10.5281/zenodo.1308993)

**License:**
The original study and its associated materials are distributed under the terms of the **Creative Commons Attribution License**, which permits unrestricted use, distribution, and reproduction in any medium, provided the original authors and source are credited.


## Dataset Preprocessing
### Create the subset.csv
Unlike the original study, the primary dataset used for training and validation in this project was pre-filtered to include only patient records from **Departments A and B**. Furthermore, due to hardware constraints, a 40% random sample of this filtered data was extracted (`subset.csv`) to serve as our main training set. For the final out-of-domain evaluation, a separate dataset containing records exclusively from **Department C** was used as the test set (`1stproject-TestSet.csv`).

```python
df=pd.read_csv('1stproject.csv')
df_subset= df.sample(frac=0.4,random_state=1)
df_subset.to_csv('subset.csv',index=False) 
```

## Modeling Approach

My strategy was conducted in two main phases to build and evaluate two distinct ensemble models. Initially, I developed a **Soft Voting Classifier** as an experimental baseline. This allowed me to understand how different base learners (experts) perform on specific feature subsets and how they interact through a simple weighted average. 

Building upon the insights gained from this initial experiment, I subsequently designed and a **Stacking Classifier**. The stacking approach was utilized to further optimize predictive performance by allowing a meta-learner to intelligently combine the probabilities of the base models, rather than relying on static weights.

Below is a detailed analysis of each architecture and its specific implementation.

### 1. Soft Voting Classifier

For the initial modeling phase, I implemented a **Soft Voting Ensemble** utilizing a "Domain Expert" architecture. Instead of feeding all features indiscriminately into a single model, I semantically grouped the data to train specialized base learners.

**Data Preprocessing & Feature Selection:**
The raw data was split into Train (70%), Validation (15%), and Test (15%) sets using a stratified approach. Missing numeric values were imputed with the median, while categorical features were imputed with the most frequent value and subsequently One-Hot Encoded. To reduce dimensionality and remove zero-variance predictors, a `VarianceThreshold` was applied. Following this, a `RandomForestClassifier` was used to identify and isolate the **Top 100 most important features**.

**The "Domain Expert" Architecture:**
The selected 100 features were parsed and separated into four distinct semantic groups using keyword matching. Each group was then assigned to a specific algorithmic "expert":
* **Vitals Expert (`LGBMClassifier`):** Trained exclusively on physiological measurements (e.g., pulse, blood pressure, O2).
* **Medications Expert (`LogisticRegression`):** Focused on patient medication history. This subset was additionally scaled using `StandardScaler` to accommodate the linear model.
* **Labs Expert (`CatBoostClassifier`):** Dedicated to laboratory test results (mins, maxes, medians).
* **History Expert (`CatBoostClassifier`):** Handled the remaining medical history and demographic features.

**Hyperparameter Tuning & Optimization:**
I utilized **Optuna** for Bayesian Optimization to simultaneously tune:
* The learning rates, depths, and regularization parameters of the individual base learners.
* The voting weights (`w_vit`, `w_med`, `w_lab`, `w_his`) assigned to each expert in the meta-ensemble.

The optimization objective was a custom scoring function based on the **F2-Score**, which heavily penalizes False Negatives (missed admissions). However, to ensure clinical viability, the function strictly penalized the model if Precision dropped below a 0.60 threshold.

**Final Inference:**
The final tuned Voting Classifier outputs the weighted average probability from all experts. The decision threshold was fine-tuned on the validation set to maximize the F2-Score before making the final Admit/Discharge predictions on the test set. All necessary components (model, preprocessor, feature lists, and threshold) were serialized as `soft_voting_artifacts.pkl` for seamless inference.

[![Soft Voting Training Results](plots/soft_voting_eval.png)](plots/soft_voting_eval.png)


The Soft-Voting Ensemble demonstrates **excellent predictive power (AUC = 0.91)** and is heavily optimized for clinical safety. By fine-tuning the model to maximize the F2-Score and applying a conservative decision threshold (0.34), the ensemble successfully prioritizes patient safety by minimizing dangerous False Negatives while maintaining an acceptable level of precision.

### 1. Global Predictive Power
* **ROC Curve (AUC = 0.91):** An Area Under the Curve of 0.91 indicates outstanding general separability. This means the model assigns a higher risk probability to a true "Admit" patient than a "Discharge" patient in 91% of cases.
* **Precision-Recall Curve (Val):** The PR curve remains robust before declining, which is a critical indicator for imbalanced medical datasets. It demonstrates that the model maintains solid precision even when pushed to maximize recall.

### 2. Clinical Safety & Decision Logic
The model’s success is anchored in its specialized threshold logic:
* **Optimized Threshold (0.34):** Rather than using a default 0.50 cutoff, the threshold was lowered to 0.34 to prioritize safety. Any patient with a >34% probability of admission is flagged.
* **High Recall (0.89):** The model successfully identified **3,966 out of 4,447** true admissions, missing only 481. This 89% sensitivity is vital for emergency triage where missed admissions (False Negatives) carry high risk.
* **Controlled Precision (0.59):** The model achieves high safety by over-triaging, with a precision of 0.59. In clinical settings, this F2-optimized trade-off (0.8093) is highly acceptable, as observation is safer than premature discharge.

### 3. Model Stability & Generalization
* **Learning Curve:** The convergence of the training and validation lines suggests that the model has effectively learned the underlying patterns without excessive memorization.
* **Bias-Variance Decomposition:**
    * **Variance (0.027):** The extremely low variance confirms the ensemble is highly stable and not overly sensitive to noise in the training data.
    * **Bias (0.163):** Bias is the primary source of error, showing that the model's errors are systematic. This is expected in heavily regularized ensembles, ensuring better robustness on unseen real-world data.

### 4. Expert Contribution
The **AUC per Expert** chart highlights the influence of different data domains:
* **Labs Expert:** The dominant contributor, indicating that laboratory results provide the strongest signal for admission.
* **Vitals & Meds Experts:** Provide strong secondary signals, ensuring physiological measurements drive the decision process.
* **History Expert:** Offers the weakest individual signal, confirming that objective clinical data is more predictive than historical demographics in this triage context.


### 2. Stacking Classifier 


To improve upon the static weights of the Soft Voting model, I designed a **Stacking Classifier**. This advanced architecture uses a meta-learner to intelligently figure out the best way to combine the predictions of multiple diverse base models during the training phase.

**Feature Selection & Clinical Engineering:**
Following the standard preprocessing (imputation and One-Hot Encoding), I used a `LGBMClassifier` on a subset of the training data to select the **Top 100 features** based on Information Gain. To specifically combat False Negatives (admitted patients misclassified as safe to discharge), I engineered **14 custom clinical features** (`fe_*`). These features target specific high-risk patterns identified in the data, such as patients with borderline ESI scores (Level 3) combined with severe abdominal pain, and subtle metabolic stress indicators.

**Feature Subspacing (Model Diversity):**
To force the base learners to learn different patterns and prevent them from memorizing the exact same signals, I utilized **Feature Subspacing**. I created 5 distinct feature sets from the training data. Each set contained:
1. The top 5 "anchor" features (the absolutely most important ones).
2. A randomized subset of the remaining 95 features.
3. **All 14 custom engineered clinical features** (guaranteed to be seen by every model).

**The Ensemble Architecture & Training:**
These 5 feature sets were fed into 5 distinct, powerful base learners:
* Model 1: `LGBMClassifier`
* Model 2: `RandomForestClassifier`
* Model 3: `CatBoostClassifier`
* Model 4: `CatBoostClassifier` (Alternative initialization)
* Model 5: `XGBClassifier`

The probability outputs of these 5 models (generated via cross-validation during the stacking process) were then fed into a **Meta-Learner (`LogisticRegression`)**, which learned how to optimally combine their predictions.

**Hyperparameter Tuning:**
I used **Optuna** to optimize the learning rates across the gradient boosting models (LGBM, CatBoost, XGBoost) and the meta-learner simultaneously. The optimization target was a custom F-beta score (**F1.5-Score**) evaluated on a proxy validation set, a metric specifically chosen to lean heavily toward Recall (safety) while maintaining a respectable Precision.

**Final Training Artifacts:**
After finding the optimal hyperparameters, the final Stacking Classifier was trained on the entire preprocessed training set. The complete pipeline (the trained model, the preprocessor, and the specific feature lists used) was serialized and saved as `stacking_model_artifacts.pkl` to be used later for inference.

![Stacking Training Results](plots/stacking_eval.png)

## Stacking Training & Evaluation Analysis

The **Stacking Classifier** represents the most advanced stage of this project, achieving a high level of predictive sophistication with an **AUC of 0.92**. By using a meta-learner to combine diverse base models, this approach significantly refines the decision-making process compared to the baseline.

### 1. Superior Predictive Performance
* **ROC Curve (AUC = 0.92):** The stacking architecture slightly outperforms the soft-voting model in overall discrimination. The meta-learner effectively filters the noise from base learners, resulting in a more precise risk estimation.
* **Precision-Recall Curve (Train):** The PR curve shows exceptional stability, maintaining near-perfect precision for a large portion of the recall range. This indicates that the stacking model is highly confident when identifying high-risk admissions.

### 2. High-Precision Triage Logic
* **Optimized Threshold (0.40):** The stacking model utilizes a 0.40 threshold, which is higher than the soft-voting baseline. This reflects a more confident model that requires less "aggressive" threshold lowering to maintain safety.
* **Balanced Classification:**
    * **Admit Precision (0.86):** A standout metric. Unlike the soft-voting model, 86% of the patients flagged for admission by the Stacker were actually admitted. This significantly reduces "alarm fatigue" in a clinical setting.
    * **Recall (0.67):** While the recall is lower than the soft-voting baseline (due to the F1.5 optimization vs F2), the model identifies **2,981 out of 4,447** admissions with much higher accuracy, minimizing unnecessary hospital resource utilization.
* **Total Accuracy (0.86):** The overall accuracy of 86% demonstrates that the meta-learner provides a very robust and reliable classification across both classes.

### 3. Error Analysis & Robustness
* **Learning Curve:** The curves for training and validation show a very healthy trend. The gap is narrowing consistently as data increases, proving that the **Feature Subspacing** technique effectively prevented the complex stacking model from overfitting.
* **Bias-Variance Decomposition:**
    * **Variance (0.040):** Slightly higher than the soft-voting model, which is expected given the higher complexity of a stacked ensemble. However, it remains well within safe limits for a production-ready model.
    * **Bias (0.180):** The bias is stable, indicating that the meta-model has found a strong "middle ground" between the varying opinions of the five base experts.

### 4. Expert Synergy (Subspacing Success)
The **AUC per Expert (g1-g5)** chart reveals the success of the feature subspacing strategy:
* **Uniform Performance:** All five "subspace experts" show nearly identical performance (AUC ~0.85-0.87).
* **Meta-Learner Advantage:** The fact that the final stacking AUC (0.92) is significantly higher than any individual expert (max 0.87) proves that the **Logistic Regression meta-learner** is successfully extracting unique insights from each subspace rather than just picking the "best" model. This confirms that the engineered features and randomized subspaces provided truly complementary information.

## Workflows

### 1. Soft Voting Classifier

![Soft Voting Workflow](model_logic_diagrams/Soft_Voting_Workflow.png)

### 2. Stacking Classifier (Advanced Ensemble)

![Stacking Workflow](model_logic_diagrams/Stacking_Workflow.png)

## 3. Final Evaluation: Soft Voting vs. Stacking

To conclusively determine the best approach, I developed a unified inference script that evaluates both the **Soft Voting Classifier** and the **Stacking Classifier** side-by-side on the completely unseen test set (`1stproject-TestSet.csv`, representing Department C). 

This script acts as the ultimate benchmark, ensuring a fair and rigorous comparison of their generalization capabilities.

### The Unified Inference Pipeline
The evaluation process runs both models through their respective paces:

1. **Artifact Loading:** The script dynamically loads the saved artifacts (`.pkl` files) for both models, including their specific preprocessors, label encoders, feature lists, and optimized thresholds.
2. **Soft Voting Path:** The raw test data is transformed using the standard preprocessing pipeline, subset to the Top 100 features, and evaluated against the globally optimized validation threshold.
3. **Stacking Path (Advanced):** The test data goes through standard preprocessing, followed by the rigorous extraction of the 14 custom clinical engineered features (`fe_*`). 

### Specialized Logic for the Stacking Model
Because the Stacking model was designed to handle complex clinical realities, two additional steps were applied exclusively to its probability outputs before final classification:
* **Prior Correction:** The raw probabilities were mathematically recalibrated. This adjusts the model's perspective from the artificially balanced training environment (30% admit rate) to match the real-world expected prevalence (15% admit rate).
* **4-Tier False Negative Reduction:** I applied a custom clinical override logic. This logic dynamically lowers the admission threshold for specific high-risk patient profiles (e.g., severe abdominal pain with borderline ESI, or highly frail elderly patients), acting as a safety net against dangerous False Negatives.

### Visual Dashboard & Metrics
To easily digest the results, the script generates a comprehensive **Comparison Dashboard** (`plots/model_comparison.png`) that highlights:
* **ROC Curves (AUC):** Evaluating the overall ability of both models to separate Admits from Discharges.
* **Admit-Class Focus:** A direct comparison of Precision, Recall, and the custom F-beta (F1.5/F2) scores, focusing specifically on the critical "Admit" class.
* **Probability Distributions:** Histograms showing how confidently each model separates the true classes around the decision threshold.
* **Confusion Matrices:** A granular view of True Positives, True Negatives, False Positives, and False Negatives to understand exactly where each model excels or fails.

![Model Comparison Dashboard](plots/model_comparison.png)
## Final Model Comparison: Soft Voting vs. Stacking (Test Set)

The final evaluation on the out-of-domain test set (Department C) revealed distinct behaviors for each architecture. While the **Soft Voting** model prioritized raw safety (Recall), the **Stacking Classifier** demonstrated superior precision and overall reliability in a realistic clinical environment.

It is also important to consider the class distribution differences between the training and testing environments. The training data (Departments A and B) contained approximately **30% admissions and 70% discharges**, while the out-of-domain test set (Department C) had a much more imbalanced distribution of roughly **15% admissions and 85% discharges**. Therefore, some degree of performance degradation and reduced generalization is relatively expected, since the models were exposed during training to a substantially different patient outcome distribution.

### 1. ROC Analysis & Discriminative Power
* **ROC Curve:** The **Soft Voting Classifier (AUC = 0.867)** shows higher overall discriminative ability on the test set compared to the **Stacking model (AUC = 0.824)**. This suggests that the Soft Voting ensemble is slightly more effective at ranking patients by risk across the entire population.

* **Probability Distributions:**
  * The **Soft Voting** distribution is more spread out, requiring a very low threshold (0.34) to capture admissions.
  * The **Stacking** distribution is highly skewed toward zero, indicating it is much more "selective." Even with a 0.40 threshold, it remains very conservative in its predictions.

### 2. The Precision-Recall Trade-off (Admit Class)
The comparison of **Admit Class Metrics** highlights the core difference in philosophy between the two models:

* **Recall (Safety):** Soft Voting achieves a higher **Recall (0.82)** compared to Stacking (0.77). It is more effective at catching admissions but at a significant cost to efficiency.

* **Precision (Efficiency):** Stacking shows superior **Precision (0.41)** compared to Soft Voting (0.37). This means the Stacking model produces fewer "false alarms," which is critical for reducing hospital overcrowding and staff burnout.

* **F2-Score:** Both models perform similarly in terms of the F2-Score (~0.60), showing they both maintain a good balance between safety and precision, though Stacking leads slightly in overall F1-performance (0.538 vs 0.508).

### 3. Confusion Matrix Analysis
* **Soft Voting:** Correctly identified **9,174 admissions** but generated **15,772 false positives**. It is a "safety-first" model that over-triages heavily to ensure few patients are missed.

* **Stacking:** Identified **8,639 admissions** with significantly fewer false positives (**12,282**). By reducing false alarms by nearly 3,500 cases compared to Soft Voting, the Stacking model proves to be a more efficient tool for resource management.

### 4. Overall Model Behavior
* The **Soft Voting** model demonstrates a more aggressive prediction strategy, prioritizing sensitivity and minimizing missed admissions, even at the expense of a large number of false positives.

* The **Stacking Classifier** exhibits a more balanced and conservative behavior, maintaining strong recall while substantially reducing unnecessary admission predictions. This suggests better calibration and improved practical usability in real-world clinical triage settings.

* Overall, the results indicate that although both models remain robust under distribution shift, the **Stacking approach generalizes more effectively** to Department C by offering a better trade-off between patient safety and operational efficiency.