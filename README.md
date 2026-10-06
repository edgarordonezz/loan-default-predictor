# Loan Default Predictor

Predicting whether a loan applicant will default, using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset (307,511 applicants, 122 features).

The focus of this project is careful data work on a messy, heavily imbalanced dataset: auditing missing values, figuring out *why* data is missing, engineering missingness flags, and choosing metrics and thresholds that make sense for a lending problem instead of optimizing accuracy.

## Results

Only about 8% of applicants default, so accuracy is misleading (predicting "no default" for everyone scores ~92%). Models are compared on ROC-AUC and on precision/recall for the default class.

| Model | ROC-AUC | Precision (default) | Recall (default) |
| --- | --- | --- | --- |
| Logistic Regression, unscaled (baseline) | 0.614 | 0.12 | 0.53 |
| Logistic Regression, scaled | **0.749** | 0.16 | 0.68 |
| Logistic Regression, scaled, F1-optimal threshold (0.671) | 0.749 | 0.241 | 0.397 |
| Random Forest (100 trees, max depth 10) | 0.741 | 0.17 | 0.63 |

All models use `class_weight='balanced'` and an 80/20 stratified train/test split.

**Takeaways**

- Scaling features fixed a convergence failure in Logistic Regression and raised ROC-AUC from 0.614 to 0.749.
- Random Forest performed about the same as Logistic Regression (0.741 vs 0.749), and both models agreed on 7 of their top 15 features. That points to the data (severe imbalance, limited features) being the ceiling, not the model choice.
- The F1-optimal threshold (0.671) gives a balanced reference point, but trades away a lot of recall. In lending, a missed defaulter usually costs more than a rejected good applicant, so a lower, recall-leaning threshold is likely the better business choice.

## Approach

### 1. Class imbalance

- 91.9% of applicants repaid, 8.1% defaulted.
- Used `class_weight='balanced'` so the model can't ignore the minority class, and evaluated with ROC-AUC, precision, and recall instead of accuracy.

### 2. Missing data audit

67 of 122 columns had missing values. Instead of dropping or filling blindly, each group of missing columns was investigated: is the missingness random, and does it predict default on its own?

| Column(s) | Missing | Finding | Handling |
| --- | --- | --- | --- |
| 48 columns, mostly building/apartment stats | over 40% | Correlation with TARGET under 0.05 | Dropped |
| `EXT_SOURCE_1` | over 40% | Strong predictor (corr -0.155) despite missingness; pensioners ~2.5x overrepresented in missing rows | Kept, median fill + `EXT_SOURCE_1_missing` flag |
| `EXT_SOURCE_3` | 19.8% | Missingness not tied to income type; missing rows default more (9.3% vs 7.8%) | Median fill (left-skewed) + `EXT_SOURCE_3_missing` flag |
| `OCCUPATION_TYPE` | 31.3% | 57% of missing rows are pensioners, but 43% are working | Filled with `"Unknown"` rather than guessing "Retired" |
| 6 `AMT_REQ_CREDIT_BUREAU_*` columns | 41,519 rows, all the same rows | Missing rows default more (10.3% vs 7.7%) | Filled with 0 (no bureau record) + `bureau_missing` flag |
| 4 `*_CNT_SOCIAL_CIRCLE` columns | 1,021 rows, all the same rows | Missing rows default *less* (3.5% vs 8.1%) | Filled with 0 + `social_circle_missing` flag |
| `NAME_TYPE_SUITE` | 0.4% | Small, categorical | Filled with most common value |
| 5 small numeric columns | under 0.3% | Trivial | Median fill |

### 3. Anomaly: `DAYS_EMPLOYED`

- 55,374 rows had `DAYS_EMPLOYED = 365243` (about 1,000 years), a placeholder value.
- 99.96% of those rows were pensioners, so the value means "not currently employed."
- Replaced with 0 and added a `DAYS_EMPLOYED_anomaly` flag. Those applicants default less often (5.4% vs 8.7%).

### 4. Encoding

- One-hot encoding for low-cardinality categorical columns, so the model doesn't assume an order between categories.
- Label encoding for the two high-cardinality columns (`OCCUPATION_TYPE`, `ORGANIZATION_TYPE`) to avoid exploding the feature count.

### 5. Modeling

- Logistic Regression baseline, then `StandardScaler` (fit on the training set only) to fix non-convergence.
- Random Forest for comparison, since it can capture non-linear relationships and doesn't need scaling.
- Precision-recall curve analysis and an F1-optimal threshold search instead of relying on the default 0.5 cutoff.
- Feature importance comparison between the two models.

## Project structure

```
loan-default-predictor/
├── 01_eda_baseline.ipynb   # EDA, preprocessing, models, and analysis
└── .gitignore              # excludes the dataset (*.csv) and model files (*.pkl)
```

## Running it

1. Download `application_train.csv` from the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk/data) and place it in the project root. The dataset is not included in the repo.
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Open and run the notebook:
   ```
   jupyter notebook 01_eda_baseline.ipynb
   ```

## Next steps

- Merge in additional Home Credit tables, starting with `bureau.csv`, to give the models more signal.
- Try gradient boosting (XGBoost or LightGBM).
- Move imputation and encoding into a scikit-learn `Pipeline` so every preprocessing step is fit on the training data only.
- Choose a final threshold based on an explicit cost of false negatives vs false positives.

## Tech

Python, pandas, NumPy, scikit-learn, Matplotlib, seaborn, Jupyter
