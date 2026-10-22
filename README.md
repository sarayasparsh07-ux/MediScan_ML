# MediScan | Diagnostic Intelligence Lab

The group's MediScan project as two deliverables built on **one shared analysis**:

- **`MediScan_Full_Project.ipynb`**: the team notebook (Student Guide Parts A–E plus extensions), executed with all outputs visible.
- **The Streamlit dashboard** (`app.py`): the same results, interactive, across ten pages.

Both use the same code path and seeds. The notebook's appendix **asserts** that every leaderboard number, combination
result, threshold, calibration score and recommendation is identical in both.

**Team** (five members; the brief suggests 3–4 and the professor approved a fifth verbally)

| Member | Model(s) built |
|---|---|
| Sparsh | Random Forest + Decision Tree |
| Aradhana | SVM (linear kernel) |
| Shashaank | SVM (RBF kernel) |
| Shirley | Gaussian Naive Bayes |
| Mahi | Logistic Regression |

> Educational modelling exercise. Results come from a constructed teaching dataset and are **not** evidence that
> any model is clinically validated.

---

## 1. Run it

```bash
cd mediscan_dashboard
pip install -r requirements.txt          # exact versions, Python 3.13
python -m mediscan.cache                 # optional: confirms the bundled analysis cache loads (or builds it)
streamlit run app.py                     # dashboard → http://localhost:8501
jupyter notebook MediScan_Full_Project.ipynb
```

- **Instant start.** `requirements.txt` pins the exact versions the bundled `.cache/` was built with, so the dashboard
  opens in seconds. If an exact pin cannot be installed, use `pip install -r requirements-flexible.txt`; the app then
  recomputes the analysis once (about 2 minutes) and caches it. Run `python -m mediscan.cache` **before a demo** to do
  that in advance. The cache is keyed to the CSV and library versions; deleting `.cache/` is always safe.
- The notebook takes about 3 minutes top to bottom. Run it **from this folder** so the appendix can cross-check the
  dashboard (in Colab it still runs; the cross-check is skipped).

**Tests** (optional): `python -m pytest -q` → 37 tests (pipeline correctness, statistics, combination engine,
every page and interactive control headless, Start Here route, calibration controls, team names).

## 2. Project structure

```
mediscan_dashboard/
├── MediScan_Full_Project.ipynb  # executed team notebook (summary, Parts A–H, summary check, appendix)
├── app.py                       # Streamlit UI — 10 pages, sidebar navigation
├── data/mediscan.csv            # supplied dataset (800 × 10)
├── mediscan/
│   ├── config.py                # features, model specs, team, palette (single source of truth)
│   ├── data.py                  # robust CSV location + 12 start-up validation checks
│   ├── pipeline.py              # split, scaling, six models, metrics, rankings, experiments
│   ├── ensembles.py             # 57-candidate combination search, thresholds, fresh-fold check, calibration
│   ├── stats.py                 # McNemar exact test, corrected resampled t-test, single-model thresholds
│   ├── recommendation.py        # transparent decision rules + professor-ready paragraph
│   ├── charts.py                # Plotly figures in the MediScan visual identity
│   ├── cache.py                 # fingerprinted disk cache (python -m mediscan.cache to warm it)
│   └── qa.py                    # 33 live QA checks (shown in-app AND enforced by tests)
├── tests/                       # test_pipeline.py, test_app.py, screenshot_pages.py
├── .streamlit/config.toml       # theme
├── .cache/                      # precomputed analysis (auto-rebuilt if anything differs)
├── requirements.txt             # exact pins (instant start)
└── requirements-flexible.txt    # version ranges (one-time recompute)
```

## 3. Dashboard pages

**Start Here** is the presentation route: both recommendations, five headline numbers, and five tabs in speaking order
(① the ring → ② linear vs RBF boundary → ③ scaling → ④ combinations → ⑤ recommendation), each with a chart, three
talking points and a pointer to the full page.

Then: Dataset & EDA · Model Leaderboard · Confusion Matrix Lab · PCA & Decision Boundaries · Scaling & Feature
Insights · Significance & Thresholds (McNemar, corrected t-test, threshold tuning, **calibration**) · Model
Combinations · Clinical Recommendation · Methodology (one tab per model, labelled with its builder).

Every section shows its chart or table plus **one takeaway**; method explanations sit in collapsed "ℹ️" expanders.

## 4. Protocol

| Step | Setting |
|---|---|
| Predictors | 8 measurements; `patient_id` and `diagnosis` excluded |
| Split | `train_test_split(test_size=0.25, random_state=1, stratify=y)` → 600 train / 200 test (93 malignant) |
| Scaling | `StandardScaler` fitted on training rows only; refit inside every CV fold |
| Positive class | 1 = Malignant for every class-specific metric |
| Six required models | exact Student Guide settings (LR, Gaussian NB, linear SVM, RBF SVM C=3, tree depth 6, forest 200) |
| Thresholds | probability of malignancy (SVMs Platt-calibrated); grid 0.05…0.95; chosen on training out-of-fold predictions only |
| Clinical cost | 10 × missed cancer + 1 × false alarm (illustrative assumption) |
| Combinations | 6 singles + 50 soft votes (2–4 models) + 1 stack; stage 1 tunes & shortlists 8 (+ Random Forest); stage 2 re-scores on 10 × 5-fold fresh CV (seed 2026) |
| Calibration | reliability diagram (10 equal-width bins), Brier score, expected calibration error; out-of-fold and test |

## 5. Results

**Six required models (held-out test set, n = 200)**

| Model (built by) | Accuracy | Recall (M) | Precision (M) | Missed (FN) | False alarms (FP) |
|---|---|---|---|---|---|
| Logistic Regression (Mahi) | 73.0% | 68.8% | 71.9% | 29 | 25 |
| Naive Bayes (Shirley) | 96.0% | **98.9%** | 92.9% | **1** | 7 |
| SVM linear (Aradhana) | 72.5% | 68.8% | 71.1% | 29 | 26 |
| SVM RBF (Shashaank) | 97.0% | 96.8% | 96.8% | 3 | 3 |
| Decision Tree (Sparsh) | **97.5%** (tie) | 96.8% | **97.8%** | 3 | 2 |
| Random Forest (Sparsh) | **97.5%** (tie) | 97.8% | 96.8% | 2 | 3 |

**Recommendations (identical in notebook and dashboard)**

- **Best of the six required models: Random Forest** (default threshold).
- **Recommended configuration (extension): soft vote of NB + SVM-RBF + RF at threshold 0.40.** On fresh CV it averages
  1.7 missed cancers and 10.4 false alarms per 600 patients, vs 2.2 / 17.8 for Random Forest at 0.40, and is cheaper
  in 9 of 10 repeats. On the test set: 1 of 93 missed, 7 false alarms.
- **How sure:** the gain is fewer false alarms at the same cancer-catching rate (false alarms p = 0.057, missed cancers
  p = 0.80, corrected t-tests). No shortlisted alternative is significantly worse, so **Random Forest at 0.40** is a
  statistically comparable, simpler fallback.
- **Most cautious alternative:** NB + DT + RF at 0.30 (0.6 missed and 21.8 false alarms per 600; equivalent cost).

## 6. Findings worth flagging

1. **The 8-feature PCA shows the ring only partially** (29.8% of variance kept); the crisp ring is in Cell Size × Cell Shape.
2. **On two features the linear SVM predicts "benign" for everyone**: 53.5% is the benign share, recall 0%.
3. **Gaussian Naive Bayes succeeds because its boundary is an ellipse** (malignant cell variance about 7× benign).
4. **Only the RBF-SVM is scale-sensitive**: 61.5% → 97.0%; unscaled, Marker Level supplies 95% of patient distances.
5. **The four curved models are close to tied.** McNemar separates none of them on the test set, while every
   linear-vs-curved comparison is significant. In cross-validation the only significant gaps are Random Forest over
   the Decision Tree (recall and accuracy).
6. **At the default 0.5 threshold no combination beats Random Forest**; combinations win only once thresholds are tuned.
7. **Selecting on the same folds flatters the winner**: the stage-1 leader (4-model vote, cost 20) scores 30.6 on fresh folds.
8. **Ranking is not calibration.** Random Forest (ROC-AUC 0.996) has squeezed probabilities (ECE 0.14); the ensemble is
   better (0.09); the Platt-calibrated RBF-SVM (0.016) and the stack (0.024) are well calibrated. Thresholds were set
   from observed errors, so the recommendation is unaffected.
9. **Backpropagation is not required** for any of the six models; an optional MLP shows no clear gain.

## 7. Quality assurance

- 12 start-up data checks; 33 live QA checks (Methodology page), all passing.
- 37 automated tests, including independent re-implementation of the Student Guide, McNemar/Holm/corrected-t-test
  against known values, exact VotingClassifier equivalence, threshold-rule correctness, and a headless run of every
  page and control.
- The notebook executes end to end with no errors. Its **summary check** re-derives all 18 claims in the one-screen
  summary, and its **appendix** asserts equality with the dashboard (including calibration).
- Every page was reviewed in headless Chromium at desktop (1440 px) and phone (400 px) width, including each
  Start Here tab, for clipped or overlapping labels.

**Limitations:** one 200-patient test set (one patient ≈ 1.1 points of recall); constructed data with no external
validation cohort; Random Forest's and the ensemble's probabilities are not calibrated risks; the cost ratio is
illustrative; the fresh folds reuse the same 600 training patients, so they guard against one lucky split but are not
external validation. On phone-width screens, long chart titles can be shortened by the browser (a Plotly limitation);
the presentation layout is designed for a laptop or projector.
