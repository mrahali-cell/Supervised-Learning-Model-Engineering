# Project Findings Log: Adult / Census Income

Supervised Learning & Model Engineering (MSA-DATB09-04).
Working notes for the final report (about 10 pages) and the oral defense
(7 min + 3 min Q&A). Last updated 2026-10-18 (after the fairness audit).

Every number comes from the EDA notebook (`notebooks/01_eda.ipynb`) or from the
baseline run (`python -m src.train`, 5-fold CV on the development set, test set
untouched). Items marked **PROVISIONAL** must be re-checked before use.

To read this file nicely in VS Code: open it, then press `Ctrl+Shift+V`
(Markdown preview).

---

## 1. The project in short

- **Dataset:** Adult / Census Income (OpenML, version 2; 1994 US Census;
  48,842 rows, 15 columns). State this at the top of the report.
- **Business case** (latest version, replaces the earlier public-benefit
  outreach case; team confirmation pending): a company builds ML models that
  predict whether a person earns more or less than $50K. Its clients use them
  for purposes such as audits. If a model is biased, the audit decisions can be
  unfair, and the clients can face sanctions or legal action and lose money.
  Our study measures the bias of such a model and tests a mitigation, so the
  company and its clients can deliver fairer predictions.
- **Deliverable:** a model card with performance by group, the fairness
  findings, one mitigation, and what it costs in accuracy (S8, S9).
- **Target:** `high_income` = 1 if income > $50K (23.94% of rows). It was
  `low_income` = 1 in the first EDA version (76.06%). Flipped in `src/data.py`
  on branch `soumaya/dev` (uncommitted; team agreement pending) because under
  this case the orientation is a free choice and the rare class as positive
  fits the course (S6): PR-AUC is informative (random baseline 0.24, not 0.76),
  class weights and SMOTE can be tested as taught, and it matches the dataset
  documentation, the literature and the MIT module. The train/test rows are
  exactly the same as before the flip (the split is stratified on the old
  column).
- **Costly error:** no single one, because the downstream uses vary. The
  exposure is unequal treatment, so we report error rates for BOTH classes,
  by group.
- **Key metrics:** ROC-AUC and PR-AUC for ranking; Brier score and ECE for
  probability quality, overall and by group; per group: selection rate,
  recall (TPR), FPR and precision (PPV).
- **Fairness definition** (proposed; fix it BEFORE seeing results, Session 9):
  primary = **equalized odds** (equal TPR and equal FPR for women and men),
  because for a general-purpose model both kinds of error can harm someone.
  Legal screen = the **four-fifths rule** on selection rates (the S9 example of
  disparate-impact guidance). Also report demographic parity and predictive
  parity, and explain why they conflict.

---

## 2. Findings log

Each finding lists where it can be used: **R** = report section, **S** = slide.

### F1 · Data: a 1994 snapshot, no timestamp, no person ID
- **Evidence:** EDA Steps 1 and 6. 48,842 rows; only "reasonably clean" records
  (age > 16, income > $100, hours > 0).
- **Course link:** S1 leakage and splitting.
- **Use in:** R framing / limitations. S data slide.

### F2 · Data: missing values are limited and not random
- **Finding:** workclass and occupation go missing for the same 2,799 people;
  10 "Never-worked" people have occupation missing because it does not apply.
- **Evidence:** EDA Step 2. workclass 2,799 (5.73%), occupation 2,809 (5.75%),
  native-country 857 (1.75%).
- **Course link:** S5 imputation inside the Pipeline.
- **Use in:** R data cleaning. S backup.

### F3 · Data: 52 duplicates removed before the split
- **Finding:** 52 exact duplicate rows (0.11%) were dropped before the split,
  so the same record cannot appear in both train and test.
- **Evidence:** EDA Step 2; 48,842 -> 48,790 rows after cleaning.
- **Course link:** S1 group leakage.
- **Use in:** R leakage audit.

### F4 · Leakage: capital gains dropped
- **Finding:** capital-gain and capital-loss were dropped. A short
  questionnaire does not ask for them, and they are probably built from the
  income itself.
- **Evidence:** EDA Step 6. Capital gain >= $1: 61.7% earn > $50K; >= $5,000:
  90.1%; >= $10,000: 98.4%; the 244 people at the 99,999 cap: 100%. 91.7% of
  capital-gain values are 0.
- **Course link:** S1 target leakage: "could we know this at prediction time?"
- **Use in:** R leakage audit (key argument). S leakage slide.

### F5 · Our scores must NOT be compared with published ones
- **Finding:** published results of 0.92-0.93 AUC used the capital gains we
  excluded on purpose. Our baseline is about 0.89 ROC-AUC.
- **Evidence:** literature (section 7) vs our baseline (section 3).
- **Course link:** S1 honest evaluation.
- **Use in:** R results / limitations. S say it out loud in the Q&A.

### F6 · Features: three more columns dropped
- **Finding:** `relationship` (Husband / Wife) reveals sex; `education`
  duplicates `education-num`; `fnlwgt` is a sampling weight with no signal.
- **Evidence:** EDA Steps 3 and 4c. All 19,716 Husbands and 2,331 Wives are
  married. fnlwgt medians: 178,811 (<= $50K) vs 176,729 (> $50K).
- **Course link:** S9 proxy variables.
- **Use in:** R features. S backup.

### F7 · Features: protected columns are audit-only
- **Finding:** `sex`, `race` and `native-country` are never model inputs; they
  are kept only for the fairness audit.
- **Evidence:** `src/data.py`: `FEATURES` (6 inputs) and `SENSITIVE`.
- **Course link:** S9 audit by group.
- **Use in:** R fairness. S fairness slide.

### F8 · EDA: base rates differ strongly between groups
- **Finding:** 30.4% of men vs 10.9% of women earn > $50K (about 3 to 1). Race
  groups range from 11.7% to 26.9%.
- **Evidence:** EDA Step 4c. Asian-Pac-Islander 26.9%, White 25.4%, Other
  12.3%, Black 12.1%, Amer-Indian-Eskimo 11.7%.
- **Course link:** S9 base rates, and why fairness definitions conflict.
- **Use in:** R fairness (core argument). S fairness slide.

### F9 · EDA: strongest signals for income
- **Finding:** marital status, occupation, education, workclass, age.
- **Evidence:** EDA Step 4c. Married-civ-spouse 44.6% > $50K vs 4.5-10.1% for
  other statuses; Exec-managerial 47.8% vs Priv-house-serv 1.2%; education-num
  9 -> 15.9%, 13 -> 41.3%, 15 -> 74.0%.
- **Course link:** S8 sanity check for SHAP later.
- **Use in:** R EDA. S EDA slide.

### F10 · Split: locked test set
- **Finding:** development set 39,032 rows, test set 9,758 rows, both 23.94%
  high income; 5-fold stratified CV on the development set. The test set stays
  locked until the final evaluation.
- **Evidence:** `python -m src.data` output.
- **Course link:** S1 and S5 honest evaluation.
- **Use in:** R protocol. S protocol slide.

### F11 · Split: group sizes for the fairness audit (checked in EDA Step 7b)
- **Finding:** the split is stratified on the target only, not on sex or race.
  Sex groups are large enough in both parts. Two race groups are too small to
  audit; the others are usable with wide margins. No restratification needed.
- **Evidence:** high earners in the test set: 356 women and 1,980 men (a rate
  such as recall carries about ±5 points for women). Test-set race groups:
  Amer-Indian-Eskimo 87 and Other 80 people (too small), Asian-Pac-Islander 311,
  Black 899, White 8,381.
- **Course link:** S8 slice size and uncertainty.
- **Use in:** R limitations.

### F12 · Features: engineered columns
- **Finding:** fixed rules inside the Pipeline: `is_married`, `has_degree`,
  `part_time`, `overtime`, `age_squared`.
- **Evidence:** development set, % earning <= $50K: married 55.5% vs not married
  93.6%; degree (education-num >= 13) 52.0% vs 83.9%; hours < 35 about 93%,
  hours = 40 78.7%, hours > 40 56-65%; age <= 24 98.9%, 45-54 60.6%, 65+ 79.6%.
- **Course link:** S5 Pipeline and leakage-safe features.
- **Use in:** R feature engineering. S features slide.

### F13 · Features: the medians hid the hours signal
- **Finding:** the EDA said hours-per-week has "no signal" because both medians
  are 40, but the rate of low income changes clearly by hours regime.
- **Evidence:** same numbers as F12 (hours buckets).
- **Course link:** S8 "headline numbers hide structure".
- **Use in:** R feature engineering. S a good "we corrected ourselves" point.

### F14 · Models: simple model nearly as good as complex ones
- **Finding:** all baseline models rank people well (ROC-AUC about 0.89), and
  they are close to each other.
- **Evidence:** section 3. ROC-AUC: Logistic Regression 0.889 (engineered) vs
  Random Forest 0.890 vs LightGBM 0.892; PR-AUC 0.708 vs 0.711 vs 0.716 (random
  baseline 0.239). Fold standard deviation about 0.005 (ROC-AUC) and 0.013
  (PR-AUC), so the gaps are within noise.
- **Course link:** S2-S4 model progression.
- **Use in:** R model comparison. S results slide.

### F15 · Models: engineered features help only Logistic Regression
- **Finding:** Logistic Regression improves from 0.883 to 0.889 ROC-AUC and
  from 0.688 to 0.708 PR-AUC. Random Forest and LightGBM find the same
  patterns by themselves (no change).
- **Evidence:** section 3 (base vs engineered).
- **Course link:** S3-S4 trees capture non-linearity and interactions; S2
  linear models do not.
- **Use in:** R model comparison. S results slide.

### F16 · Imbalance: class weights did not help (first look)
- **Finding:** `class_weight="balanced"` leaves the ranking metrics and the best
  F1 essentially unchanged and makes the probabilities worse (Brier score).
  Re-weighting moves the scores, but the same effect is available by moving the
  cutoff, without damaging calibration.
- **Evidence:** Brier 0.110 -> 0.137 (LightGBM) and 0.114 -> 0.143 (Logistic
  Regression, base features); ROC-AUC unchanged; PR-AUC 0.716 -> 0.716
  (LightGBM) and 0.688 -> 0.684 (Logistic Regression); best F1 0.672 -> 0.671
  (LightGBM, engineered). Default parameters only: re-check after tuning, and
  test SMOTE before concluding.
- **Course link:** S6 remedy hierarchy (threshold, class weights, SMOTE last)
  and S7 (calibration cost).
- **Use in:** R imbalance (answers requirement 3). S imbalance slide.

### F17 · Metrics: PR-AUC is informative with the rare class as positive
- **Finding:** random ranking scores 0.24; the models score 0.69-0.72. This
  separates the models better than ROC-AUC does and shows how much room is
  left. It was not informative with the first target orientation (baseline
  0.76, all models 0.96).
- **Evidence:** Dummy PR-AUC 0.239; Logistic Regression 0.688-0.708; Random
  Forest 0.711-0.712; LightGBM 0.716.
- **Course link:** S6 ROC-AUC vs PR-AUC and the prevalence baseline.
- **Use in:** R metrics. S results slide.

### F19 · Tuning: Optuna adds very little (PROVISIONAL)
- **Finding:** tuning improves the models by a few thousandths, much less than
  the fold-to-fold noise, and the order stays the same: Logistic Regression <
  Random Forest < LightGBM. The simple model is still within about 0.011 PR-AUC
  of the best one.
- **Evidence:** section 3b. Mean 5-fold CV PR-AUC after 25 trials per model:
  Logistic Regression 0.708 (no gain vs engineered default 0.708), Random
  Forest 0.715 (default 0.711), LightGBM 0.719 (default 0.716); fold standard
  deviation about 0.013. LightGBM's best score plateaus around 0.718-0.719 from
  trial 9 on, so a bigger budget is unlikely to help much.
- **Caveat:** these are CV scores of the best trial, selected on the same folds,
  so they are slightly optimistic (S5). The honest number is the single test-set
  evaluation at the end (or nested CV).
- **Course link:** S5 hyperparameter search; CV optimism.
- **Use in:** R tuning. S results slide.

### F20 · Tuning: the imbalance remedy is mostly not needed, and costs calibration
- **Finding:** when class weighting was part of the search space, Optuna did
  not find a strong re-weighting useful for ranking (none for Logistic
  Regression, mild for LightGBM). The Random Forest that chose balanced class
  weights has a clearly worse Brier score, which fits the S7 cost of
  re-weighting (not isolated here: to confirm by refitting it without weights).
- **Evidence:** Logistic Regression chose `class_weight=None`; LightGBM chose
  `scale_pos_weight` = 1.32 (the S6 heuristic would be about 3.2); Random Forest
  chose `balanced_subsample` and its CV Brier is 0.136 vs 0.111 for Logistic
  Regression and LightGBM.
- **Course link:** S6 remedy hierarchy, S7 calibration.
- **Use in:** R imbalance + calibration. S imbalance slide.

### F21 · Tracking: every study and trial is logged in MLflow
- **Finding:** 78 runs in experiment `adult-income-bias-audit`: 3 parent runs
  (one per model) and 75 nested trial runs (25 each). Each trial logs its
  hyperparameters, CV metrics and a per-fold metrics file; each parent logs the
  best parameters and the optimisation-history figure.
- **Evidence:** `python -m src.train tune`; browse with
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Runs are stored in
  `mlflow.db` and `mlruns/` (both git-ignored); best parameters are in
  `reports/best_params.json`.
- **Course link:** project requirement 7 (experiment tracking).
- **Use in:** R tracking (take screenshots). S backup.

### F22 · Calibration: Random Forest very over-confident, LightGBM mildly, Logistic Regression fine
- **Finding:** the tuned Random Forest says "0.91" for people of whom only 81%
  really are high earners; its curve sits far below the diagonal. LightGBM is
  slightly over-confident. Logistic Regression is already calibrated. The Random
  Forest's problem is probably its balanced class weights (see F20), not the
  algorithm itself: not isolated here.
- **Evidence:** section 3c. Raw ECE / Brier: Logistic Regression 0.006 /
  0.1113; Random Forest 0.132 / 0.1365; LightGBM 0.030 / 0.1112. Highest
  probability bin, predicted vs observed: Random Forest 0.912 vs 0.811;
  LightGBM 0.835 vs 0.813; Logistic Regression 0.817 vs 0.804. Figure:
  `reports/figures/calibration_diagnosis.png`.
- **Course link:** S7 reliability diagram, Brier, ECE, over-confidence.
- **Use in:** R calibration. S calibration slide.

### F23 · Calibration: Platt and isotonic fix it without hurting the ranking much
- **Finding:** recalibration brings the Random Forest and LightGBM to the level
  of Logistic Regression. Platt keeps the ranking (monotone map); isotonic gives
  the lowest ECE but its step function creates ties and costs a little PR-AUC.
- **Evidence:** section 3c. Random Forest ECE 0.132 -> 0.004 (Platt) / 0.002
  (isotonic), Brier 0.1365 -> 0.1104. LightGBM ECE 0.030 -> 0.008 / 0.001,
  Brier 0.1112 -> 0.1096 / 0.1097, PR-AUC 0.7181 -> 0.7178 (Platt) / 0.7124
  (isotonic). Figure: `reports/figures/calibration_before_after.png`.
- **Caveat:** each calibrator is scored by a second cross-validation on
  out-of-fold predictions (never on the rows it was fitted on), but the base
  models were tuned on these same folds, so the numbers are slightly
  optimistic. The test-set check is still to do.
- **Course link:** S7 Platt vs isotonic; "calibration changes probability
  quality, not ranking".
- **Use in:** R calibration. S calibration slide.

### F24 · Calibration by sex: raw probabilities over-predict, more for women in relative terms
- **Finding:** before recalibration the Random Forest and LightGBM predict
  higher probabilities than the observed rates for both sexes. In relative terms
  the gap is larger for women, so a raw score does not mean the same thing for
  women and men. After
  recalibration both groups are close to the observed rate. The Brier score is
  lower for women (0.06 vs 0.135) only because their base rate is lower, not
  because of better calibration.
- **Evidence:** mean predicted vs observed rate (women 0.109, men 0.304).
  LightGBM raw: women 0.132, men 0.337; after Platt 0.112 and 0.302. Random
  Forest raw: women 0.204, men 0.455; after Platt 0.108 and 0.304. Logistic
  Regression: 0.114 and 0.301 (already fine). Figure:
  `reports/figures/calibration_by_sex.png`.
- **Course link:** S7 "check calibration by segment"; S9 predictive parity is
  calibration by group.
- **Use in:** R calibration + fairness. S fairness slide.

### F25 · Fairness audit (before): unequal error rates and selection, equal meaning of a flag
- **Finding:** with one group-blind threshold, the model finds far fewer of the
  high-earning women than of the high-earning men (62% vs 81%), flags women much
  less often overall (11.8% vs 41.7%), and wrongly flags far fewer lower-earning
  women (5.6% vs 24.5%). Yet a flag means almost the same thing for both sexes
  (precision 0.58 vs 0.59): predictive parity roughly holds, equalized odds and
  the four-fifths rule do not. This is the S9 conflict in practice, driven by
  base rates of 10.9% (women) vs 30.4% (men).
- **Evidence:** section 3d, best-F1 threshold 0.321, LightGBM + Platt,
  out-of-fold. Gaps (women minus men): TPR -0.191, FPR -0.189, selection rate
  -0.299, PPV -0.015; four-fifths ratio 0.28 (the screen asks for >= 0.8). 95%
  bootstrap intervals: TPR gap [-0.218, -0.165], FPR gap [-0.196, -0.181],
  selection gap [-0.307, -0.291], ratio [0.268, 0.296]: the gaps are not noise.
- **Course link:** S9 demographic parity, equalized odds, predictive parity,
  four-fifths rule, base-rate theorem; S8 uncertainty of slice metrics.
- **Use in:** R fairness (core). S fairness slide.

### F26 · Mitigation: equalized odds closes the error-rate gaps, at a modest cost, and widens the precision gap
- **Finding:** per-group thresholds (Fairlearn `ThresholdOptimizer`,
  post-processing) bring the TPR and FPR of women and men together. The price is
  about 3 points of balanced accuracy, and a flag now means less for women than
  for men. The four-fifths ratio improves but still fails.
- **Evidence:** section 3d, compared with the group-blind threshold that
  maximises the same objective (balanced accuracy). Equalized-odds gap 0.228 ->
  0.006; balanced accuracy 0.811 -> 0.782; F1 0.658 -> 0.622; recall 0.859 ->
  0.825; precision 0.534 -> 0.499. PPV gap -0.057 -> -0.296. Selection-rate gap
  -0.327 -> -0.111; four-fifths ratio 0.337 -> 0.743. The optimizer is
  cross-fitted on out-of-fold scores, so it is not scored on the rows it was
  fitted on.
- **Course link:** S9 post-processing and the fairness-accuracy trade-off.
- **Use in:** R fairness (before / after). S fairness slide.

### F27 · The three definitions cannot all be satisfied (measured)
- **Finding:** enforcing equal opportunity or demographic parity instead of
  equalized odds moves the other gaps in different directions.
- **Evidence:** section 3d. Equal opportunity: TPR gap -0.001 but FPR gap
  -0.116, PPV gap -0.177, ratio 0.52, balanced accuracy 0.804 (only -0.007).
  Demographic parity: selection gap +0.001 and ratio 1.00 (passes the four-fifths
  rule) but TPR gap +0.179, FPR gap +0.099, PPV gap -0.343, balanced accuracy
  0.770 (-0.041). Passing the legal screen therefore costs the error-rate
  parity we prioritised.
- **Course link:** S9 "fairness definitions conflict when base rates differ".
- **Use in:** R fairness (discussion). S fairness slide.

### F28 · Race: gaps exist too, and small groups are unstable
- **Finding:** at the same threshold, Black people are selected about half as
  often as White people and have a lower TPR; Asian-Pac-Islander people look
  like White people. Two groups are too small for reliable metrics.
- **Evidence:** section 3d. Selection rate Black 16.0% vs White 33.7%; TPR 0.697
  vs 0.788; FPR 0.084 vs 0.184; Asian-Pac-Islander 36.6% / 0.795 / 0.213.
  Amer-Indian-Eskimo (n = 383) and Other (n = 326) are below 500 people in the
  development set. Base rates: White 25.4%, Asian-Pac-Islander 26.3%, Black
  12.4%. Only sex was mitigated.
- **Course link:** S8 slice size, S9 groups.
- **Use in:** R fairness (limitations). S backup.

### F29 · Caveat: per-group thresholds are a policy decision
- **Finding:** `ThresholdOptimizer` uses the person's sex at decision time, and
  for equalized odds it can randomise decisions, so two similar people may get
  different outcomes. Some legal contexts do not allow explicit use of a
  protected attribute when deciding (S9). An in-processing method
  (`ExponentiatedGradient`) needs sex only for training, not for predicting; we
  have not run it.
- **Evidence:** `src/fairness.py`; S9 mitigation slide. How much randomisation
  the fitted thresholds use has not been measured yet.
- **Use in:** R fairness (discussion, limits of use). S Q&A.

### F30 · EDA: the model inputs are proxies for sex
- **Finding:** marital status, occupation and hours worked differ so much
  between women and men that a model can rebuild much of `sex` from them, even
  though `sex` is not an input. Leaving a protected column out does not make a
  model blind to it, so fairness has to be measured on the outputs, by group.
- **Evidence:** EDA Step 4d. Married-civ-spouse 60.9% of men vs 15.3% of women;
  Adm-clerical 25.3% of women vs 5.9% of men; Craft-repair 2.2% vs 18.6%;
  average hours 36.4 (women) vs 42.4 (men).
- **Course link:** S9 proxy variables ("fairness through unawareness").
- **Use in:** R fairness + EDA. S fairness slide.

### F31 · EDA: where the income gap between women and men comes from
- **Finding:** much of the overall gap (30.4% vs 10.9% earning > $50K) comes
  from who is married: among married people the rates are equal. Among unmarried
  people, and within education and hours groups, a gap remains. These are
  descriptive associations, not causes.
- **Evidence:** EDA Step 4d, % earning > $50K, women vs men: married 44.8% vs
  44.6%; not married 4.8% vs 8.2%; bachelor or more 26.3% vs 57.1%; below
  bachelor 6.6% vs 20.9%; under 35 h 6.2% vs 7.7%; 35-40 h 10.2% vs 26.2%; over
  40 h 21.6% vs 44.1%.
- **Course link:** S8 slices; S9 historical bias and base rates.
- **Use in:** R EDA + fairness (why the gap exists in the data). S backup.

### F18 · Operating point: the cutoff trades wrong flags against misses
- **Finding:** a stricter cutoff flags fewer people and is right more often,
  but finds fewer of the high earners. With no single costly error in this case,
  the neutral choice is the best-F1 point (LightGBM: precision 0.59, recall
  0.78, FPR 0.17). A client must choose its own cutoff for its own use.
- **Evidence:** section 4 table (LightGBM, out-of-fold).
- **Course link:** S6 threshold = operating point.
- **Use in:** R results (key trade-off). S results slide.

---

## 3. Baseline results

Default parameters, 5-fold CV, development set, positive class = `high_income`
(base rate 0.239). Mean +/- standard deviation over the 5 folds. The last four
columns are the best-F1 operating point, pooled out-of-fold (cutoff picked on
the same predictions, so slightly optimistic: use it to compare models, not as a
final result). **PROVISIONAL** until tuned.

| Model | Features | ROC-AUC | PR-AUC | Brier | Best F1 | Precision | Recall | FPR |
|---|---|---|---|---|---|---|---|---|
| Dummy (always the majority) | n/a | 0.500 | 0.239 | 0.182 | 0.386 | 0.239 | 1.000 | 1.000 |
| Logistic Regression | base | 0.883 ± 0.006 | 0.688 ± 0.012 | 0.114 | 0.659 | 0.575 | 0.773 | 0.180 |
| Logistic Regression | engineered | 0.889 ± 0.006 | 0.708 ± 0.014 | 0.111 | 0.668 | 0.595 | 0.762 | 0.164 |
| Logistic Regression (balanced) | base | 0.883 ± 0.006 | 0.684 ± 0.012 | 0.143 | 0.659 | 0.570 | 0.783 | 0.186 |
| Logistic Regression (balanced) | engineered | 0.889 ± 0.006 | 0.706 ± 0.014 | 0.140 | 0.667 | 0.585 | 0.776 | 0.173 |
| Random Forest | base | 0.891 ± 0.006 | 0.712 ± 0.014 | 0.111 | 0.673 | 0.596 | 0.772 | 0.165 |
| Random Forest | engineered | 0.890 ± 0.005 | 0.711 ± 0.013 | 0.111 | 0.672 | 0.587 | 0.787 | 0.174 |
| LightGBM | base | 0.892 ± 0.005 | 0.716 ± 0.012 | 0.110 | 0.672 | 0.590 | 0.780 | 0.171 |
| LightGBM | engineered | 0.892 ± 0.005 | 0.716 ± 0.012 | 0.110 | 0.672 | 0.591 | 0.778 | 0.169 |
| LightGBM (balanced) | base | 0.892 ± 0.005 | 0.716 ± 0.012 | 0.137 | 0.670 | 0.605 | 0.752 | 0.155 |
| LightGBM (balanced) | engineered | 0.892 ± 0.005 | 0.716 ± 0.012 | 0.137 | 0.671 | 0.597 | 0.767 | 0.163 |

---

## 3b. Tuned results (Optuna)

Protocol: one Optuna study per model (TPE sampler, seed 42), 25 trials each,
same 5 stratified folds, objective = mean CV PR-AUC of the full Pipeline
(engineered features included). Class weighting was part of the search space.
The test set was not used. Mean +/- standard deviation over the 5 folds, for the
best trial of each study (slightly optimistic, see F19).

| Model | CV PR-AUC | CV ROC-AUC | CV Brier | Best parameters |
|---|---|---|---|---|
| Logistic Regression | 0.708 ± 0.014 | 0.889 ± 0.006 | 0.111 | C = 5.18, L2, no class weights |
| Random Forest | 0.715 ± 0.014 | 0.891 ± 0.006 | 0.136 | 300 trees, max_depth 15, min_samples_leaf 10, max_features log2, balanced_subsample |
| LightGBM | 0.719 ± 0.013 | 0.893 ± 0.005 | 0.111 | 350 trees, learning_rate 0.016, num_leaves 27, min_child_samples 36, subsample 0.69, colsample 0.92, scale_pos_weight 1.32 |

Reference (defaults, engineered features, section 3): PR-AUC 0.708 / 0.711 /
0.716 and Brier 0.111 / 0.111 / 0.110 for the same three models.

---

## 3c. Calibration of the tuned models (development set, out-of-fold)

Predictions are out-of-fold (5 folds, best parameters of section 3b).
Calibrated rows come from a second 5-fold CV on those predictions. ECE uses 10
equal-frequency bins. "Top bin" = the 10% of people with the highest predicted
probability. Run: `python -m src.evaluate calibration`. **PROVISIONAL**
(tuned on the same folds; test-set check pending).

All people:

| Model | Variant | Brier | ECE | PR-AUC | ROC-AUC | Top bin predicted | Top bin observed |
|---|---|---|---|---|---|---|---|
| Logistic Regression | raw | 0.1113 | 0.0060 | 0.7080 | 0.8887 | 0.817 | 0.804 |
| Logistic Regression | Platt | 0.1114 | 0.0060 | 0.7076 | 0.8887 | 0.816 | 0.804 |
| Logistic Regression | isotonic | 0.1115 | 0.0026 | 0.7030 | 0.8877 | 0.807 | 0.805 |
| Random Forest | raw | 0.1365 | 0.1323 | 0.7140 | 0.8912 | 0.912 | 0.811 |
| Random Forest | Platt | 0.1104 | 0.0041 | 0.7137 | 0.8912 | 0.812 | 0.811 |
| Random Forest | isotonic | 0.1104 | 0.0021 | 0.7081 | 0.8902 | 0.814 | 0.811 |
| LightGBM | raw | 0.1112 | 0.0299 | 0.7181 | 0.8927 | 0.835 | 0.813 |
| LightGBM | Platt | 0.1096 | 0.0076 | 0.7178 | 0.8927 | 0.801 | 0.811 |
| LightGBM | isotonic | 0.1097 | 0.0014 | 0.7124 | 0.8919 | 0.814 | 0.811 |

By sex (mean predicted probability vs observed rate; observed: women 0.109,
men 0.304):

| Model | Variant | Women predicted | Men predicted | ECE women | ECE men |
|---|---|---|---|---|---|
| Logistic Regression | raw | 0.114 | 0.301 | 0.015 | 0.011 |
| Random Forest | raw | 0.204 | 0.455 | 0.094 | 0.151 |
| Random Forest | Platt | 0.108 | 0.304 | 0.011 | 0.006 |
| LightGBM | raw | 0.132 | 0.337 | 0.023 | 0.034 |
| LightGBM | Platt | 0.112 | 0.302 | 0.009 | 0.012 |
| LightGBM | isotonic | 0.113 | 0.302 | 0.010 | 0.007 |

Reading: the best candidate so far is **LightGBM + Platt** (lowest Brier,
ranking kept, ECE 0.008, both sexes close to their observed rates).
Logistic Regression is simpler and already calibrated but about 0.011 PR-AUC
lower. Not final: the fairness audit and the test-set check come first.

---

## 3d. Fairness audit by sex (development set, out-of-fold)

Model: tuned LightGBM + Platt scaling. Predictions are out-of-fold; the test set
is not used. Definitions fixed before looking at the results: primary =
equalized odds; legal screen = four-fifths rule; also reported: demographic
parity, equal opportunity, predictive parity. Mitigation = one technique,
`ThresholdOptimizer` (per-group thresholds), cross-fitted. Run:
`python -m src.fairness audit`. Figure: `reports/figures/fairness_by_sex_dev.png`.
**PROVISIONAL** (development set only; the before / after on the test set is
still to do).

Women vs men at the single best-F1 threshold (0.321):

| Group | n | Base rate | Selection rate | TPR (recall) | FPR | PPV |
|---|---|---|---|---|---|---|
| Women | 12,917 | 0.109 | 0.118 | 0.621 | 0.056 | 0.577 |
| Men | 26,115 | 0.304 | 0.417 | 0.812 | 0.245 | 0.591 |

Decision rules compared (all people). The "before" row for the mitigation is the
group-blind threshold that maximises balanced accuracy (same objective as the
ThresholdOptimizer); the best-F1 row is the neutral operating point of section 4.

| Decision rule | Accuracy | Balanced acc. | F1 | Precision | Recall |
|---|---|---|---|---|---|
| One threshold (best F1) | 0.817 | 0.806 | 0.672 | 0.589 | 0.783 |
| One threshold (best balanced accuracy) | 0.787 | 0.811 | 0.658 | 0.534 | 0.859 |
| Equalized odds (ThresholdOptimizer) | 0.760 | 0.782 | 0.622 | 0.499 | 0.825 |
| Equal opportunity (ThresholdOptimizer) | 0.787 | 0.804 | 0.653 | 0.535 | 0.837 |
| Demographic parity (ThresholdOptimizer) | 0.765 | 0.770 | 0.613 | 0.505 | 0.780 |

Fairness gaps (women minus men; four-fifths ratio = women / men, screen: >= 0.8):

| Decision rule | Selection gap | TPR gap | FPR gap | Equalized-odds gap | PPV gap | Four-fifths ratio |
|---|---|---|---|---|---|---|
| One threshold (best F1) | -0.299 | -0.191 | -0.189 | 0.191 | -0.015 | 0.282 |
| One threshold (best balanced accuracy) | -0.327 | -0.143 | -0.228 | 0.228 | -0.057 | 0.337 |
| Equalized odds (ThresholdOptimizer) | -0.111 | 0.006 | -0.003 | 0.006 | -0.296 | 0.743 |
| Equal opportunity (ThresholdOptimizer) | -0.213 | -0.001 | -0.116 | 0.116 | -0.177 | 0.521 |
| Demographic parity (ThresholdOptimizer) | 0.001 | 0.179 | 0.099 | 0.179 | -0.343 | 1.002 |

By race at the best-F1 threshold (groups under 500 people are unstable):

| Race | n | Base rate | Selection rate | TPR | FPR | PPV |
|---|---|---|---|---|---|---|
| White | 33,333 | 0.254 | 0.337 | 0.788 | 0.184 | 0.594 |
| Black | 3,784 | 0.124 | 0.160 | 0.697 | 0.084 | 0.540 |
| Asian-Pac-Islander | 1,206 | 0.263 | 0.366 | 0.795 | 0.213 | 0.571 |
| Amer-Indian-Eskimo (small) | 383 | 0.110 | 0.188 | 0.667 | 0.129 | 0.389 |
| Other (small) | 326 | 0.123 | 0.156 | 0.700 | 0.080 | 0.549 |

---

## 4. The cutoff trade-off (LightGBM, engineered features, out-of-fold)

Reading: of everyone the model flags as high income, at least the "precision"
share really is. **Recall** = share of truly high earners found. **FPR** = share
of lower earners flagged by mistake. Flagging everyone gives precision 0.24. The
cutoffs were chosen on the same out-of-fold predictions they are scored on, so
these numbers are slightly optimistic. A client picks the row that fits its use;
for our neutral comparison we use the best-F1 point (precision 0.59, recall
0.78, FPR 0.17).

| Minimum precision | Recall (high earners found) | FPR (lower earners flagged) | Score cutoff | Share of people flagged |
|---|---|---|---|---|
| 0.50 | 0.894 | 0.281 | 0.176 | 42.8% |
| 0.60 | 0.758 | 0.159 | 0.341 | 30.2% |
| 0.70 | 0.582 | 0.079 | 0.503 | 19.9% |
| 0.80 | 0.349 | 0.027 | 0.695 | 10.5% |

---

## 5. Decisions log

| Decision | Who / when | Status |
|---|---|---|
| Dataset: Adult / Census Income (OpenML v2) | Soumaya, from the project brief | Confirmed |
| Business case: model vendor whose income-prediction models are used for audits; study = find and reduce bias to limit legal and financial exposure | Soumaya, latest revision (replaces public-benefit outreach) | Chosen; team confirmation pending. EDA Step 5 still says "energy discount program (preferred)" |
| Target orientation: income > $50K = 1 (`high_income`) | Proposed by Claude, applied by Soumaya on `soumaya/dev`, uncommitted | Open: team agreement; EDA notebook text still says `low_income` |
| Original target `low_income` = 1 for income <= $50K; key metric recall on low earners | Manel (EDA) | Replaced on `soumaya/dev` by `high_income` (see next row); `main` still has the original |
| 6 model inputs; sex, race, native-country excluded; capital-gain/loss, fnlwgt, education, relationship dropped | Manel (EDA) | In the code; team confirmation pending |
| 80/20 stratified split, `random_state=42`, 5-fold stratified CV | Manel (`src/data.py`) | In the code, merged in main |
| Calibration protocol: out-of-fold predictions on the development set, Platt and isotonic scored by a second 5-fold CV, reliability by sex; test set not used | Soumaya with Claude | Done in `src/evaluate.py` (`python -m src.evaluate calibration`), uncommitted |
| EDA notebook aligned with the audit case: new Steps 4d (groups, proxies), 5 (framing), 7b (group sizes) and 8 (bias sources, hand-over); target shown as `high_income` | Soumaya with Claude, 2026-10-19 | Done in `notebooks/01_eda.ipynb` (re-executed), uncommitted. Based on the earlier 29-cell version restored on `soumaya/dev`; the Step 9 preview of the newer version is not in it |
| Fairness protocol: sex as the audited attribute; equalized odds primary and four-fifths rule as legal screen (fixed before the results); one mitigation, `ThresholdOptimizer`, cross-fitted on out-of-fold scores; race only audited | Soumaya with Claude | Done on the development set in `src/fairness.py` (`python -m src.fairness audit`), uncommitted; test-set evaluation pending |
| Candidate final model: LightGBM + Platt scaling | Proposed by Claude from F19-F24 | Provisional: confirm after the fairness audit and the single test-set evaluation |
| Tuning protocol: Optuna (TPE, seed 42), 25 trials per model, objective = mean 5-fold CV PR-AUC; class weighting is a tuned hyperparameter; MLflow logging in a SQLite file | Soumaya with Claude | Done in `src/train.py` (`python -m src.train tune`), uncommitted |
| Engineered features: is_married, has_degree, part_time, overtime, age_squared | Soumaya with Claude, 2026-10-18 | Uncommitted on the local checkout |
| Cutoff: best-F1 point for the neutral comparison; clients choose their own | Proposed | Open: confirm with the team |
| Fairness definition: equalized odds as primary, four-fifths rule as legal screen | Proposed | Open: confirm before looking at fairness results |

---

## 6. Open questions and things to verify

- [ ] Is the business case final for all three members? Ask Manel and Hadiqa;
      update EDA Step 5 wording.
- [ ] Team agreement on the flipped target (`high_income`): `main` and the EDA
      notebook still use `low_income`; src/data.py, README and this log use the
      new one.
- [ ] Which cutoff do we show in the report (best F1 or a client-driven choice)
      and how do we justify it?
- [x] Are the sex and race groups big enough in the test set for a stable
      fairness audit? Sex: yes. Race: two groups (87 and 80 people) are too
      small, so race is audited descriptively for them (F11).
- [ ] The tuned models do NOT separate (gaps of 0.004-0.011 PR-AUC vs a fold
      standard deviation of 0.013). How do we choose the final model? Options:
      paired comparison across folds, nested CV (S5), and secondary criteria
      (calibration, interpretability, fairness behaviour).
- [ ] Re-run the tuning with a larger budget or nested CV? (LightGBM plateaus
      early, so the expected gain is small.)
- [x] Are the probabilities calibrated? Diagnosed on the development set (F22,
      F23, F24). Still to do: fit the final calibrator on all out-of-fold
      predictions and check the calibrated final model on the test set, once.
- [ ] Is the Random Forest's over-confidence caused by its balanced class
      weights? Refit it without weights and compare the ECE (F20, F22).
- [ ] Where does the model fail (weakest segment by recall)? Not checked yet
      (Session 8).
- [x] Does a mitigation close the gaps by sex, and at what cost? On the
      development set: yes for equalized odds (gap 0.228 -> 0.006), at about 3
      points of balanced accuracy (F26). Still to do: the before / after on the
      untouched test set, once.
- [ ] Which priority do we defend in the report: equalized odds (as decided in
      advance, F26) or the four-fifths rule, which it still fails (ratio 0.74)?
      F27 shows we cannot have both.
- [ ] How much randomisation do the fitted per-group thresholds use, and is
      using sex at decision time acceptable? Consider running
      `ExponentiatedGradient` as an alternative (F29).
- [ ] The cutoffs in section 4 were chosen on out-of-fold predictions: redo
      them with a clean protocol before reporting.

---

## 7. External references used (for the report)

- **MIT OpenCourseWare, "Case Studies with Data: Mitigating Gender Bias on the
  UCI Adult Dataset"** (Nakeshimana and Najafian). Used for: gender bias framed
  as an error-rate gap; data-based fixes (drop gender, balance counts,
  counterfactual augmentation); model comparison across families. How our work
  differs: single split, default parameters, accuracy as headline, informal
  fairness definitions in the module.
  <https://ocw.mit.edu/courses/res-ec-001-exploring-fairness-in-machine-learning-for-international-development-spring-2020/c99940017c0859e6e1526724dc4759b0_MITRES_EC001S19_video7.pdf>
- **A Statistical Approach to Adult Census Income Level Prediction.** Tuned
  gradient boosting about 88% accuracy and 0.93 AUC with the full feature set
  (the benchmark we should NOT compare with, see F5).
  <https://arxiv.org/pdf/1810.10076>
- **A survey of bias in Machine Learning through the prism of Statistical
  Parity for the Adult Data Set.** Gender gap in the data (about 30% of men vs
  11% of women above $50K) and disparate impact on Adult.
  <https://arxiv.org/pdf/2003.14263>
- **Retiring Adult: New Datasets for Fair Machine Learning.** Critique: the
  $50K threshold is arbitrary and limits the dataset's value as a fairness
  benchmark (use in the limitations section).
  <https://arxiv.org/pdf/2108.04884>

Course material used: Sessions 1-9 slides (framing and leakage S1;
regularization S2; trees S3; ensembles S4; Pipelines, CV, Optuna, nested CV S5;
imbalance S6; calibration S7; diagnostics and explainability S8; fairness S9).

---

## 8. Figures and tables still to produce

- [ ] Class balance bar chart and base rates by sex and race (EDA).
- [ ] Capital-gain leakage table (F4), as a small table or bar chart.
- [ ] Model comparison table with CV mean +/- std, then the same after tuning,
      then the test score (once).
- [ ] Precision-recall trade-off plot or the table of section 4 for the chosen
      model.
- [ ] Optuna optimisation history (already saved as a figure in each MLflow
      parent run) and an MLflow runs screenshot.
- [x] Reliability diagram before and after calibration, with Brier and ECE
      (`reports/figures/calibration_diagnosis.png`,
      `calibration_before_after.png`, `calibration_by_sex.png`; also logged in
      MLflow).
- [ ] Slice table (n, prevalence, recall, precision, FPR) and the weakest
      segment.
- [ ] SHAP beeswarm plus 3-4 waterfall plots; permutation importance next to
      gain importance.
- [x] Fairness by sex before and after the mitigation on the development set
      (`reports/figures/fairness_by_sex_dev.png`, section 3d; also in MLflow).
- [ ] The same before / after table and figure on the test set (once, at the
      end), plus the fairness vs performance trade-off plot.

---

## 9. Draft structure for the 7-minute talk

| Slide | Topic | Main message / source |
|---|---|---|
| 1 | Problem and business case | Who benefits, what the costly error is (section 1) |
| 2 | Data and leakage decisions | F1-F7. Capital gains dropped on purpose (F4, F5) |
| 3 | Honest protocol | F10-F11. Locked test set, CV inside the Pipeline |
| 4 | Features and model progression | F12-F15. Simple model nearly as good; features help only the linear model |
| 5 | Imbalance and operating point | F16-F18. Why class weights did not fit; the cutoff is a business choice |
| 6 | Trust and fairness (to complete) | Calibration, weakest segment, SHAP, fairness by sex with a mitigation and its cost |
| 7 | Limitations and take-aways | 1994 data, arbitrary threshold, historical gaps, proof of concept only |

Each member must be able to explain every slide: the individual grade is
checked in the Q&A.
