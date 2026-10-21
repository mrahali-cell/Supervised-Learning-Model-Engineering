# Project Findings Log: Adult / Census Income

Supervised Learning & Model Engineering (MSA-DATB09-04).
Working notes for the final report (about 10 pages) and the oral defense
(7 min + 3 min Q&A). Last updated 2026-10-21 (after the LightGBM
diagnosis and tuning v2, and the logistic regression tuning v2: v2 now
complete for the three models).

Every number comes from the EDA notebook (`notebooks/01_eda.ipynb`) or from the
scripts in `tuning/` (5-fold CV on the development set, test set untouched;
results saved in `tuning/results/`). Items marked **PROVISIONAL** must be re-checked before use.

To read this file nicely in VS Code: open it, then press `Ctrl+Shift+V`
(Markdown preview).

---

## 1. The project in short

- **Dataset:** Adult / Census Income (OpenML, version 2; 1994 US Census;
  48,842 rows, 15 columns). State this at the top of the report.
- **Research question:** we predict whether a person earns more than $50K
  (`class`, as `high_income`) and measure **how biased the model is by sex**.
  `sex`, `race` and `native-country` are never inputs, only used for the audit.
  Hypothesis: the model reproduces the gender gap through proxies (marital
  status, occupation, hours worked; F30, F31).
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
  by Soumaya, merged in `main` on 2026-10-19 (EDA notebook and README updated
  too), because under
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

## 1b. Who did what (for the individual Q&A)

### Manel
- **GitHub repository and workflow:** created the repository, one branch per
  member (`manel/dev`, `soumaya/dev`, `hadiqa/dev`), changes reach `main`
  through pull requests.
- **EDA** (`notebooks/01_eda.ipynb`, Steps 1-4c, 6, 7 and 9): context and
  documentation; integrity (missing values, duplicates, target imbalance);
  storage vs semantic types; ranges, outliers and rare categories; features vs
  target; leakage audit (capital gains); cleaning and split; preview of the data
  after feature engineering and preprocessing. Findings F1-F11, F33, F34.
- **`src/data.py`:** load -> clean (fixed rules from the EDA) -> 80/20 stratified
  split, 5-fold stratified CV; the `FEATURES` / `SENSITIVE` / `TARGET` constants
  every script imports (F7, F10).
- **`src/pipeline.py`, preprocessing:** imputation, scaling and one-hot encoding
  inside the Pipeline, re-fitted in each CV fold (F35). Feature engineering was
  added to it by Soumaya (F12).
- **README:** quick start for the team, architecture, how to run each stage.
- **Project architecture** (2026-10-19): `src/` as a library, `tuning/` for all
  training and tuning, notebooks only explain (section 5).
- **Random forest:** diagnosis before tuning (F32, `tuning/random_forest.py`,
  `notebooks/03_random_forest.ipynb`) and the tuning v2 plan (section 3e).

### Soumaya
- (to complete)

### Hadiqa
- **LightGBM:** diagnosis before tuning (F38, `tuning/lightgbm.py`,
  `notebooks/04_lightgbm.ipynb`): baseline train vs validation, validation
  curves (learning rate x trees, leaves, depth, leaf size, sampling, L1 / L2,
  class weight); tuning v2 with a search space built from the curves (F39,
  `suggest_params()` in `tuning/tune.py`, `tuning/results/v2/best_params.json`).

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
- **Evidence:** `python -m tuning.tune` (formerly `src/train.py`); browse with
  `mlflow ui --backend-store-uri sqlite:///mlflow.db`. Runs are stored in
  `mlflow.db` and `mlruns/` (both git-ignored); best parameters are in
  `tuning/results/v1/best_params.json`.
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

### F32 · Random forest diagnosis: overfitting is solved, the data sets the ceiling
- **Finding:** a fully grown tree memorises the training data; limiting tree
  growth is what fixes it. Once the leaves hold at least 5 people the
  train-validation gap is small, and tuning v1 adds nothing measurable. Its
  class weights made the probabilities worse without changing the ranking.
- **Evidence:** `python -m tuning.random_forest baseline`, 5-fold CV, saved in
  `tuning/results/random_forest/baseline.csv`. Train / validation PR-AUC (gap):
  one fully grown tree 0.977 / 0.444 (0.53); RF sklearn defaults 0.972 / 0.639
  (0.33); RF baseline, min_samples_leaf 5, 0.803 / 0.711 (0.09); RF tuned v1
  0.749 / 0.715 ± 0.013 (0.035). Validation Brier: baseline 0.111, tuned v1 0.136
  (`class_weight="balanced_subsample"`); ROC-AUC 0.890 vs 0.891. Confirms F19
  and points to the class weights as the cause of F20/F22 for the random forest
  (other parameters differ too: the `class_weight` validation curve isolates it).
  Splitting criterion: Gini 0.641 vs entropy 0.641 validation PR-AUC
  (reproduced in `tuning/results/random_forest/curves.csv`).
- **Next:** done on 2026-10-20: validation curves and tuning v2 (F36).
- **Course link:** S5 bias/variance, train vs validation gap; S7 calibration.
- **Use in:** R tuning (random forest). S results slide.

### F33 · Data: outliers classified, not deleted
- **Finding:** the extreme values are census caps (top-coding) or plausible
  edge cases, not errors, so every row is kept. The IQR rule cannot be used:
  with 91.7% of capital gains at 0, it would flag every non-zero gain.
- **Evidence:** EDA Step 4b. `capital-gain` = 99,999: 244 people (cap, real high
  earners); `hours-per-week` = 99: 137 (probable cap); `age` = 90: 55 (90+ recorded
  as 90); `fnlwgt` up to 1.49M (sampling weight, dropped in Step 6). Documented
  filters respected: minimum age 17, minimum hours 1.
- **Course link:** S1 data quality: understand a value before removing it.
- **Use in:** R data cleaning. S backup (Q&A: "what about outliers?").

### F34 · Data: rare categories merged with fixed rules, before the split
- **Finding:** categories with a few dozen people are merged with the closest
  category by meaning and by income rate. A rare category gives an unstable
  estimate and a one-hot column that can be missing from some CV folds. The
  merge is a fixed rule (nothing is learned from the data), so it is safe
  before the split.
- **Evidence:** EDA Steps 4b-4c and Step 7 summary (% earning > $50K).
  `workclass`: Without-pay (21) + Never-worked (10) -> No-pay (both near 0%);
  `marital-status`: Married-AF-spouse (37, 37.8%) -> Married-civ-spouse (44.6%);
  `occupation`: Armed-Forces (15, 33.3%) -> Protective-serv (31.3%);
  `native-country` (audit only): 41 countries -> United-States (89.7%) / Mexico
  (1.95%) / Other; missing stays missing. The 10 Never-worked people with no
  occupation get `occupation = "Never-worked"` (not applicable, not unknown).
- **Course link:** S1 fixed cleaning rules vs learned transformations; S5 encoding.
- **Use in:** R data cleaning.

### F35 · Preprocessing: everything that learns from the data is inside the Pipeline
- **Finding:** imputation, scaling and one-hot encoding learn values from the
  data (a median, a mean and standard deviation, a list of categories), so they
  are fitted on the training folds only, inside the Pipeline, and re-fitted in
  each CV fold. Fitting them on all the data first would leak information from
  the validation rows (preprocessing leakage).
- **Evidence:** `src/pipeline.py`, a `ColumnTransformer` with one chain per
  column type:
  numeric (`age`, `education-num`, `hours-per-week`, `age_squared`): median
  imputation, then standardisation (z-score);
  categorical (`workclass`, `marital-status`, `occupation`): missing ->
  `"Unknown"`, then one-hot encoding with `handle_unknown="ignore"`;
  engineered 0/1 columns: passed through. 6 inputs -> 11 columns after feature
  engineering -> 37 columns after preprocessing (32 without feature
  engineering), no missing value left (EDA Step 9).
- **Why these choices:** one-hot because the categories have no order (a number
  code would invent one for a linear model); `"Unknown"` keeps the information
  that the value was missing, which is not random (F2); `handle_unknown="ignore"`
  so a category absent from a training fold does not crash; scaling matters for
  regularised logistic regression and changes nothing for trees.
- **Course link:** S1 slide 21; S5 slides 9, 11, 13 (Pipeline, ColumnTransformer,
  fit on train only).
- **Use in:** R protocol. S protocol slide.

### F36 · Random forest tuning v2: same ranking as v1, trustworthy probabilities
- **Finding:** a search space built from the validation curves gives the same
  ranking as tuning v1, a small train-validation gap, and probabilities as good
  as logistic regression's, without recalibration. Every trial lands on the
  plateau: once the space is right the search method matters little, and the
  ceiling is the information in the data.
- **Evidence:** `python -m tuning.tune --version v2 --model rf` (Optuna TPE,
  seed 42, 25 trials, objective = mean 5-fold PR-AUC; fixed: 300 trees, gini,
  no class weights; searched: leaf 5-30, depth 8-20, split 2-200, max_samples
  0.3-1.0, max_features sqrt / log2 / 0.3). Best: leaf 6, depth 16, split 10,
  max_samples 0.39, sqrt (`tuning/results/v2/best_params.json`). Then
  `python -m tuning.random_forest baseline`: train 0.754, validation
  0.715 ± 0.013, gap 0.039, ROC-AUC 0.891, Brier 0.110 (v1: 0.749 / 0.715 /
  0.035 / 0.136; baseline leaf 5: 0.802 / 0.711 / 0.091 / 0.111). All 25 trials
  between 0.709 and 0.715; best of the first 10 (random) trials 0.7144, best
  overall 0.7149.
- **Validation curves** (`tuning/results/random_forest/curves.csv` and
  `curves/*.png`; one setting at a time, the others at sklearn defaults, 200
  trees): leaf best 0.714 at 10 (plateau 5-20); depth best 0.713 at 12 (depth
  4: 0.684 with gap 0.004, underfitting; unlimited: 0.642); split best 0.714 at
  100 (plateau 50-200); max_samples 0.685 at 30% vs 0.642 at 100%; max_features
  sqrt 0.642 vs all columns 0.634; criterion gini 0.642 = entropy 0.642;
  n_estimators 0.642 at 200 vs 0.643 at 400 (gap unchanged, 0.33); class_weight
  balanced: PR-AUC 0.628 and Brier 0.145 vs 0.642 and 0.130.
- **Caveat:** the best trial is selected on the same folds, so slightly
  optimistic (S5). The settings interact: alone, leaf 10 was best; with 39% of
  people per tree, Optuna prefers leaf 6.
- **Course link:** S3 trees, S4 bagging, S5 validation curves and grid /
  random / TPE search, S6-S7 class weights cost calibration.
- **Use in:** R tuning (random forest). S tuning slide.

### F37 · Logistic regression diagnosis: no overfitting, a little bias; C hardly matters
- **Finding:** no overfitting (gap 0.001) but a little bias: being linear costs
  about 0.01 PR-AUC vs the random forest and LightGBM. Regularization can only
  add bias: C is flat from 0.3 to 100 (L1 = L2), Elastic Net never beats L2,
  class weights only hurt the Brier score. Only new information helps: married
  × hours passes the rule fixed in advance (+0.0016, better in 5 of 5 folds).
- **Evidence:** `python3 -m tuning.logistic_regression baseline | curves |
  lasso | interactions | search`, 5-fold CV, results in
  `tuning/results/logistic_regression/`, explained in
  `notebooks/02_logistic_regression.ipynb`. Train / validation PR-AUC: default
  C = 1 0.7096 / 0.7082; tuned v1 (C = 5.18, L2) 0.7097 / 0.7083 ± 0.013 (gap
  0.0014), Brier 0.111; 6 inputs only 0.688. C curves: 0.7083 for C = 1-100 (L1
  and L2); at C = 0.01, 0.698 (L2) and 0.682 (L1); L1 at C = 0.0001 = Dummy
  (0.239). Lasso keeps 31 of 37 columns at C = 1 (at 0 in all folds:
  occupation Never-worked, 8 people, and `workclass_Unknown`, an exact copy of
  `occupation_Unknown`) and 6 at C = 0.001 (education, age, hours, marital
  status). Elastic Net (l1_ratio 0-1, saga): 0.7083 for every ratio at C >= 1,
  worse with more L1 at C <= 0.01. class_weight balanced: 0.7063, Brier 0.140.
  Interactions vs tuned v1: education x hours +0.0003 (4/5 folds), age x
  education 0.0000 (3/5), married x hours +0.0016 (5/5, kept). Grid, random
  and Optuna TPE (20 trials each, v1 space): all 0.7083; TPE reproduces v1
  (C = 5.18).
- **Caveat:** the interaction and the best trials are selected on the same
  folds (slightly optimistic, S5). L1 zeros on weak or duplicate columns
  depend on the solver's shuffle (the script fixes `random_state=42`), so they
  are not a stable feature selection. Married x hours combines two proxies for
  sex (F30): re-check the fairness gap if it is adopted.
- **Next:** done on 2026-10-21: tuning v2 with the space C 0.01-100 (log), L2
  fixed, class_weight None fixed (F40).
- **Course link:** S2 logistic regression, L1 / L2 / Elastic Net; S5
  bias/variance, validation curves, grid / random / TPE search; S6 class
  weights; S7 Brier score.
- **Use in:** R tuning (logistic regression). S tuning slide.

### F38 · LightGBM diagnosis: mild variance, at the ceiling; rate and trees go together, class weights only cost calibration
- **Finding:** LightGBM's defaults overfit only a little (gap 0.04) and the
  model sits at the same ceiling as the others: every validation curve peaks at
  0.716-0.718. The learning rate and the number of trees must be read
  together: the best validation is always at rate x trees of about 5 to 8, and
  more steps overfit (unlike the forest, where more trees never hurt). The gap
  can be halved for free (fewer leaves, a depth limit, larger leaves, strong
  L1 / L2 penalties): only the train score drops. `scale_pos_weight` leaves the
  ranking unchanged and damages the probabilities.
- **Evidence:** `python3 -m tuning.lightgbm baseline | curves`, 5-fold CV,
  results in `tuning/results/lightgbm/`, explained in
  `notebooks/04_lightgbm.ipynb`. Train / validation PR-AUC (gap): defaults
  0.757 / 0.7155 ± 0.011 (0.042), 6 inputs only 0.758 / 0.7155 (engineered
  features add nothing, F15); tuned v1 0.740 / 0.7190 ± 0.012 (0.022), Brier
  0.1112 vs 0.1101 at the defaults. Curves (one setting at a time, the others at
  the defaults): learning rate x trees best 0.7170 (0.03, 200), 0.7165 (0.1,
  50), 0.7163 (0.01, 800); 0.1 with 1,600 trees 0.672 (gap 0.24); 0.01 with 25
  trees 0.686, Brier 0.156; 0.3 unstable (0.585 at 1,600 trees, fold std 0.06;
  train score falls from 0.842 to 0.787; on fold 1, 17% of training scores
  within 1e-6 of 0 or 1). num_leaves 4 / 16 / 255: 0.711 (gap 0.005) / 0.7169 /
  0.694 (gap 0.18). max_depth 2: 0.709; 4: 0.7170 (gap 0.017); 8 and more = no
  limit. min_child_samples 5-500: 0.714-0.716 (gap 0.047 -> 0.024). subsample
  0.7-1.0: 0.716; 0.3: 0.712. colsample_bytree 0.4: 0.7181 (highest point,
  +0.0026 vs 1.0). reg_alpha 10: 0.7168 (gap 0.021), 100: 0.703; reg_lambda
  100: 0.7165 (gap 0.021), no effect up to 1. scale_pos_weight 1 / 1.5 / 2 /
  3.2 / 5: PR-AUC 0.7155-0.7160, Brier 0.110 / 0.114 / 0.120 / 0.138 / 0.162.
- **Caveat:** single curves around the defaults; the settings interact (v1's
  0.719 beats every single curve by combining a slow rate with sampling, within
  the noise). The rate 0.3 points vary slightly between runs (multithreading);
  all others are reproducible.
- **Next:** done on 2026-10-21: tuning v2 with a search space built from these
  curves (F39).
- **Course link:** S4 gradient boosting (learning rate, number of trees); S2
  L1 / L2 penalties; S5 bias/variance, validation curves; S6 class weights; S7
  Brier score.
- **Use in:** R tuning (LightGBM, the candidate final model). S tuning slide.

### F39 · LightGBM tuning v2: same ranking as v1, smaller gap, the best probabilities of the project
- **Finding:** a search space built from the validation curves (F38) gives the
  same ranking as tuning v1, with a smaller train-validation gap and better
  probabilities, consistently across the folds. Removing the class weight is
  what improves the Brier score. Every trial lands on the plateau: once the
  space is right, the search method matters little, and the ceiling is the
  information in the 6 inputs (as for the random forest, F36).
- **Evidence:** `python3 -m tuning.tune --version v2 --model lgbm` (Optuna TPE,
  seed 42, 25 trials, objective = mean 5-fold PR-AUC; fixed: scale_pos_weight
  1; searched: learning rate 0.01-0.1 (log), 100-1,000 trees, leaves 8-63,
  depth 3-12, leaf size 10-200 (log), subsample 0.7-1.0, colsample 0.3-1.0,
  reg_alpha 0.01-30 (log), reg_lambda 0.01-100 (log)). Best: 950 trees,
  learning rate 0.014, 12 leaves, depth 10, leaf size 46, subsample 0.80,
  colsample 0.43, reg_alpha 1.56, reg_lambda 0.74
  (`tuning/results/v2/best_params.json`, LightGBM entry; the random forest
  entry is unchanged). Then `python3 -m tuning.lightgbm baseline`: train 0.734,
  validation 0.7189 ± 0.013, gap 0.015, ROC-AUC 0.8928, Brier 0.1096 (v1:
  0.740 / 0.7190 ± 0.012 / 0.022 / 0.8928 / 0.1112; defaults: 0.757 / 0.7155 /
  0.042 / 0.8917 / 0.1101). Per fold, same folds: PR-AUC v1 ahead in 4 of 5
  folds by 0.0004-0.0011, v2 ahead in one by 0.0031; Brier v2 better in 5 of 5
  folds by 0.0014-0.0020. 24 of 25 trials between 0.7157 and 0.7189 (one at
  0.7102); best of the first 10 (random) trials 0.7180, best overall 0.7189.
- **Reading:** small trees with many slow steps. Rate x trees = 13, above the
  5-8 of the single curves (F38), because 12-leaf trees each do less than the
  default 31-leaf trees: the settings interact (S5). `max_depth` 10 does not
  bind with 12 leaves.
- **Caveat:** the best trial is selected on the same folds, so slightly
  optimistic (S5); the test set gives the honest number. `src/evaluate.py` and
  `src/fairness.py` still read v1 (`BEST_PARAMS_VERSION` in `src/config.py`)
  until v2 exists for all three models; calibration (F22-F24) and the fairness
  audit (F25-F27) must be re-run on v2 then.
- **Course link:** S4 gradient boosting; S5 validation curves, TPE search,
  CV optimism; S6-S7 class weights cost calibration.
- **Use in:** R tuning (LightGBM, the candidate final model). S tuning slide.

### F40 · Logistic regression tuning v2: identical to v1, as predicted; v2 now complete for the three models
- **Finding:** the v2 space read from the curves (F37) finds a different C on
  the same flat plateau and exactly the same scores as v1. With this run, v2
  exists for all three models, so the project switches to v2
  (`BEST_PARAMS_VERSION = "v2"` in `src/config.py`). Across the three
  families, LightGBM v2 ranks best and has the best probabilities; the
  differences are small next to the fold noise (section 3d).
- **Evidence:** `python3 -m tuning.tune --version v2 --model lr` (Optuna TPE,
  seed 42, 25 trials; fixed: L2, no class weights; searched: C 0.01-100, log).
  Best: C = 6.80 (v1: 5.18) (`tuning/results/v2/best_params.json`, logistic
  regression entry; the other two entries unchanged). Then
  `python3 -m tuning.logistic_regression baseline`: train 0.7097, validation
  0.7083 ± 0.013, gap 0.0014, ROC-AUC 0.8887, Brier 0.1113, identical to v1 to
  4 decimals. 18 of 25 trials at 0.7083, 21 of 25 at 0.7080 or more; lowest
  0.7019 (a small C).
- **Caveat:** best trial picked on the same folds (S5), negligible on a flat
  curve. The married x hours interaction (F37) is not part of v2.
- **Next:** re-run calibration (`python3 -m src.evaluate calibration`) and the
  fairness audit (`python3 -m src.fairness audit`) on v2: sections 3c and
  F22-F27 still describe v1.
- **Course link:** S2 regularization; S5 validation curves and TPE search.
- **Use in:** R tuning (logistic regression) and model choice. S tuning slide.

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

## 3d. Tuned results v2 (search spaces built from the validation curves)

Same protocol as 3b (Optuna TPE, seed 42, 25 trials, mean 5-fold PR-AUC), with
each model's search space read from its validation curves (F36, F39, F40).
Scores from each model's diagnosis script (`python3 -m tuning.<model>
baseline`), same 5 folds, train vs validation. Best parameters in
`tuning/results/v2/best_params.json`, used by `src/evaluate.py` and
`src/fairness.py` from 2026-10-21 (`BEST_PARAMS_VERSION = "v2"`).

| Model | Train PR-AUC | Val PR-AUC | Gap | Val ROC-AUC | Val Brier | Best parameters |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7097 | 0.7083 ± 0.013 | 0.0014 | 0.8887 | 0.1113 | C = 6.80, L2, no class weights |
| Random Forest | 0.7536 | 0.7149 ± 0.013 | 0.0386 | 0.8915 | 0.1104 | 300 trees, gini, no class weights, min_samples_leaf 6, max_depth 16, min_samples_split 10, max_samples 0.39, max_features sqrt |
| LightGBM | 0.7341 | 0.7189 ± 0.013 | 0.0152 | 0.8928 | 0.1096 | 950 trees, learning_rate 0.014, num_leaves 12, max_depth 10, min_child_samples 46, subsample 0.80, colsample 0.43, reg_alpha 1.56, reg_lambda 0.74, scale_pos_weight 1 |

Changes vs v1 (3b): ranking unchanged for all three (LR 0.7083 = 0.7083, RF
0.715 = 0.715, LightGBM 0.7189 vs 0.7190); Brier better for the random forest
(0.136 -> 0.110) and LightGBM (0.1112 -> 0.1096), the same for LR. No model
uses class weights in v2. Order unchanged: LR < RF < LightGBM, with LightGBM
+0.004 over the random forest and +0.011 over LR (fold std about 0.013;
no paired test between families yet).

---

## 3c. Calibration of the tuned models (development set, out-of-fold)

Predictions are out-of-fold (5 folds, best parameters of section 3b).
Calibrated rows come from a second 5-fold CV on those predictions. ECE uses 10
equal-frequency bins. "Top bin" = the 10% of people with the highest predicted
probability. Run: `python -m src.evaluate calibration`. **PROVISIONAL**
(tuned on the same folds; test-set check pending). **Computed on v1** (section
3b): to re-run on v2 (section 3d).

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

## 3e. Random forest: diagnosis and tuning v2 (Manel)

**How a random forest learns.** Each tree picks, at every node, the split that
reduces impurity the most (Gini by default, or entropy). This is local and
greedy: unlike logistic regression and LightGBM, the forest never minimises a
global log-loss, and its "probability" is the share of trees voting yes. That is
why its probabilities are the least reliable of the three (F22). Tuning works
one level above: it chooses the rules the trees must follow, by cross-validation.

**Versions** (5-fold CV, development set; results in
`tuning/results/random_forest/baseline.csv`):

| Hyperparameter | v0: sklearn defaults | Baseline | v1: Optuna, 25 trials | v2 |
|---|---|---|---|---|
| n_estimators | 100 | 300 | 300 (searched 100-400) | 300 (fixed: flat after 200) |
| max_depth | unlimited | unlimited | 15 (searched 4-24) | 16 (searched 8-20) |
| min_samples_leaf | 1 | 5 | 10 (searched 1-30) | 6 (searched 5-30) |
| min_samples_split | 2 | 2 | 2 (not searched) | 10 (searched 2-200) |
| max_features | sqrt | sqrt | log2 (sqrt, log2, 0.5) | sqrt (sqrt, log2, 0.3) |
| max_samples | 100% | 100% | 100% (not searched) | 39% (searched 30-100%) |
| criterion | gini | gini | gini (not searched) | gini (fixed: no difference) |
| class_weight | none | none | balanced_subsample | none (fixed: confirmed by the curve) |
| **Val PR-AUC** | 0.639 ± 0.009 | 0.711 ± 0.012 | 0.715 ± 0.013 | 0.715 ± 0.013 |
| **Train-val gap** | 0.333 | 0.091 | 0.035 | 0.039 |
| **Val Brier** | 0.131 | 0.111 | 0.136 | 0.110 |

**Hypotheses tested in v2** (validation curves first, then Optuna on what matters):

| Hyperparameters | Hypothesis | Confirmed if |
|---|---|---|
| min_samples_leaf, max_depth, min_samples_split | Trees still memorise a little: stronger limits help (variance) | The gap shrinks, validation stays stable |
| max_features, max_samples | More diverse trees average their errors better | Validation goes up slightly |
| class_weight | Weighting does not help the ranking and damages the probabilities | Same PR-AUC, worse Brier |
| criterion | Gini and entropy give the same forest here | Same scores (already seen, F32) |
| n_estimators | More trees never overfit; the score reaches a plateau | Flat curve after some number of trees |
| Depth / leaf size vs fairness | Deeper trees use proxy combinations (married x occupation x hours) more, so they reproduce the gender gap more | The equalized-odds gap by sex grows with depth |

**Results of the hypotheses (2026-10-20, F36):** growth limits confirmed
(plateau at leaf 5-20, depth 12, split 50-200); diversity: smaller samples help
(0.642 -> 0.685 at 30%), max_features has no clear effect; class weights
confirmed harmful (Brier 0.130 -> 0.145); criterion no difference; trees flat
after 200; the fairness hypothesis (depth vs sex gap) is not tested yet.

**Expected gain:** about ±0.01 PR-AUC. Logistic regression, the random forest
and LightGBM all stop around 0.71 with 6 inputs (F14, F19): the ceiling is the
information in the data, not the model. The value of v2 is to explain *why*
(bias vs variance) and to get better probabilities, not to win accuracy.

**Steps:** (1) `python -m tuning.random_forest curves`; (2) fix what is settled;
(3) `python -m tuning.tune --version v2 --model rf` with the new search space;
(4) compare v0, baseline, v1 and v2. Status: done on 2026-10-20 (F36); final random forest = tuned v2.

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
| Target orientation: income > $50K = 1 (`high_income`) | Proposed by Claude, applied by Soumaya | Applied in `src/data.py`, the EDA notebook and the README; merged in main 2026-10-19 |
| Original target `low_income` = 1 for income <= $50K; key metric recall on low earners | Manel (EDA) | Replaced by `high_income` (see previous row) |
| 6 model inputs; sex, race, native-country excluded; capital-gain/loss, fnlwgt, education, relationship dropped | Manel (EDA) | In the code; team confirmation pending |
| 80/20 stratified split, `random_state=42`, 5-fold stratified CV | Manel (`src/data.py`) | In the code, merged in main |
| Outliers kept: they are census caps or plausible values (F33) | Manel (EDA Step 4b) | In the EDA; nothing to apply in the code |
| Rare categories merged; native-country grouped into United-States / Mexico / Other (F34) | Manel (EDA Steps 4b-4c, `src/data.py`) | In the code, merged in main |
| Preprocessing inside the Pipeline: median + scaling for numbers, "Unknown" + one-hot for categories (F35) | Manel (`src/pipeline.py`) | In the code, merged in main |
| Calibration protocol: out-of-fold predictions on the development set, Platt and isotonic scored by a second 5-fold CV, reliability by sex; test set not used | Soumaya with Claude | Done in `src/evaluate.py` (`python -m src.evaluate calibration`), merged in main |
| EDA notebook aligned with the audit case: new Steps 4d (groups, proxies), 5 (framing), 7b (group sizes) and 8 (bias sources, hand-over); target shown as `high_income` | Soumaya with Claude, 2026-10-19 | Done, merged in main together with Manel's Step 9 preview |
| Fairness protocol: sex as the audited attribute; equalized odds primary and four-fifths rule as legal screen (fixed before the results); one mitigation, `ThresholdOptimizer`, cross-fitted on out-of-fold scores; race only audited | Soumaya with Claude | Done on the development set in `src/fairness.py` (`python -m src.fairness audit`), merged in main; test-set evaluation pending |
| Candidate final model: LightGBM + Platt scaling; logistic regression as the transparent benchmark; the random forest tuned to show bias vs variance | Proposed by Claude from F19-F24; Manel agrees (2026-10-19) | Provisional: confirm after the fairness audit and the single test-set evaluation |
| Tuning protocol: Optuna (TPE, seed 42), 25 trials per model, objective = mean 5-fold CV PR-AUC; class weighting is a tuned hyperparameter; MLflow logging in a SQLite file | Soumaya with Claude | Done (first in `src/train.py`, now `tuning/tune.py`); results in `tuning/results/v1/` |
| Engineered features: is_married, has_degree, part_time, overtime, age_squared | Soumaya with Claude, 2026-10-18 | In `src/pipeline.py`, merged in main |
| Cutoff: best-F1 point for the neutral comparison; clients choose their own | Proposed | Open: confirm with the team |
| Fairness definition: equalized odds as primary, four-fifths rule as legal screen | Team | Adopted: fixed before the audit, used in `src/fairness.py` |
| Architecture: `src/` = library only (config, data, pipeline, models, cv, evaluate, fairness); `tuning/` = every training and tuning script, results in `tuning/results/<version>/`; notebooks only explain; `src/train.py` split into `src/models.py`, `src/cv.py`, `tuning/baseline.py`, `tuning/tune.py` | Manel, 2026-10-19 | Done, merged in main |
| Random forest diagnosis and tuning v2 (section 3e): search space read from the validation curves; final random forest = tuned v2 | Manel, 2026-10-20 | Done (F32, F36); `tuning/results/v2/best_params.json` |
| Logistic regression diagnosis (F37): baseline, C curves for L1 and L2, Elastic Net and Lasso zeros, interactions, search methods; same pattern as `tuning/random_forest.py` | Soumaya with Claude, 2026-10-20 | Done: `tuning/logistic_regression.py`, `tuning/results/logistic_regression/`, `notebooks/02_logistic_regression.ipynb`; uncommitted |
| Logistic regression tuning v2 search space: C 0.01-100 (log), penalty fixed L2, class_weight fixed None; no Elastic Net | Proposed by Soumaya with Claude from the curves (F37) | Open: team agreement to edit `suggest_params()` in `tuning/tune.py`, then `python3 -m tuning.tune --version v2 --model lr`; not run |
| Interaction rule, fixed before the run: an interaction is kept only if it adds >= 0.001 mean validation PR-AUC over tuned v1 AND is better in >= 4 of 5 folds | Soumaya with Claude, 2026-10-20 | Applied: only married x hours passes (F37) |
| Add married x hours (`is_married` x `hours-per-week`) to the logistic regression Pipeline | Proposed (F37) | Open: needs a change in `src/pipeline.py` (shared) or an LR-only Pipeline, team agreement, and a fairness re-check by sex |
| LightGBM diagnosis and tuning v2 (final model) | Hadiqa | To do: `tuning/lightgbm.py` (same pattern as `tuning/random_forest.py`), `notebooks/04_lightgbm.ipynb`, then `python3 -m tuning.tune --version v2 --model lgbm`; if v2 becomes the final model, re-run calibration and the fairness audit on it |

---

## 6. Open questions and things to verify

- [ ] Is the business case final for all three members? Ask Manel and Hadiqa;
      update EDA Step 5 wording.
- [x] Team agreement on the flipped target (`high_income`): applied in
      `src/data.py`, the EDA notebook and the README, merged in main.
- [ ] Which cutoff do we show in the report (best F1 or a client-driven choice)
      and how do we justify it?
- [x] Are the sex and race groups big enough in the test set for a stable
      fairness audit? Sex: yes. Race: two groups (87 and 80 people) are too
      small, so race is audited descriptively for them (F11).
- [ ] The tuned models do NOT separate (gaps of 0.004-0.011 PR-AUC vs a fold
      standard deviation of 0.013). How do we choose the final model? Options:
      paired comparison across folds, nested CV (S5), and secondary criteria
      (calibration, interpretability, fairness behaviour). Manel's position
      (2026-10-19): the three families tie within noise, so choose on
      calibration and explainability; keep LightGBM + Platt, with logistic
      regression as the transparent benchmark.
- [ ] Re-run the tuning with a larger budget or nested CV? (LightGBM plateaus
      early, so the expected gain is small.)
- [x] Are the probabilities calibrated? Diagnosed on the development set (F22,
      F23, F24). Still to do: fit the final calibrator on all out-of-fold
      predictions and check the calibrated final model on the test set, once.
- [x] Is the Random Forest's over-confidence caused by its balanced class
      weights? Yes (F36): changing only that setting, Brier 0.130 -> 0.145;
      tuned v2 without weights has Brier 0.110. Refit it without weights and compare the ECE (F20, F22). First
      evidence in F32 (Brier 0.111 without weights vs 0.136 with); the
      `class_weight` validation curve changes only that parameter.
- [ ] Random forest v2 objective: PR-AUC only (as v1), PR-AUC with the Brier
      score or log-loss checked, or several goals at once including the
      equalized-odds gap by sex (section 3e)?
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
- [x] Random forest baseline table: train vs validation (F32,
      `tuning/results/random_forest/baseline.csv`).
- [x] Random forest validation curves
      (`tuning/results/random_forest/curves/*.png`), one per hyperparameter.
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
| 3 | Honest protocol | F10-F11, F35. Locked test set, preprocessing and CV inside the Pipeline |
| 4 | Features and model progression | F12-F15, F32. Simple model nearly as good; features help only the linear model. Three model families tie within noise: the ceiling is the data, and changing the model does not remove a bias that comes from the data |
| 5 | Imbalance and operating point | F16-F18. Why class weights did not fit; the cutoff is a business choice |
| 6 | Trust and fairness (to complete) | Calibration, weakest segment, SHAP, fairness by sex with a mitigation and its cost |
| 7 | Limitations and take-aways | 1994 data, arbitrary threshold, historical gaps, proof of concept only |

Each member must be able to explain every slide: the individual grade is
checked in the Q&A.
