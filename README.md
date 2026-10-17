# Next-Day Estimated PM2.5 Forecasting Across Time and Districts in Dhaka Division, Bangladesh

This repository contains the end-to-end data preparation, model training, and evaluation pipelines for forecasting next-day district-level $\text{PM}_{2.5}$ concentrations ($\mu\text{g/m}^3$) across all 13 districts of Dhaka Division, Bangladesh[cite: 2, 3, 4, 5].

> **Important Scientific Caveat:** The target variable (`target_pm25_ug_m3`) represents **CAMS forecast and ERA5 reanalysis-derived estimated $\text{PM}_{2.5}$**, not direct ground-monitor observations. Model metrics evaluate agreement with this reanalysis/forecast-derived target rather than physical ground-station concentration accuracy.

---

## Repository Structure

| File | Stage | Description |
| :--- | :--- | :--- |
| `pre_training_pipeline_of_forecasting.py` | **1. Pre-Training & EDA** | Inspects raw dataset schemas, normalizes timestamps to `Asia/Dhaka` (`BST`), aggregates hourly records (requiring $\ge 18$ valid hours/day), restores missing calendar dates without interpolation, engineers strictly historical lag/rolling features, generates EDA plots, and exports split CSVs. |
| `training_code_forecast.py` | **2. Baseline Training** | Performs automated feature-leakage and data-integrity audits (`feature_audit.json`, `data_audit.json`) and trains 4 models across 5 experimental splits using the verified 18-feature baseline set. |
| `evaluation_code_forecast.py` | **3. Baseline Evaluation** | Evaluates saved baseline predictions using 14-day block bootstrap ($2,000$ replicates), decomposing temporal generalization (RQ1), spatial generalization to unseen districts (RQ2), and feature/group permutation importance stability (RQ3). |
| `forecast_revised_experiments.py` | **4. Revised Training (Humidity + LODO)** | Extracts historical relative humidity from the raw dataset, adds 3 lagged/rolling humidity predictors (21 features total), reuses saved baseline `.joblib` models for exact comparison, and runs Leave-One-District-Out (LODO) cross-validation across all 13 districts. |
| `forecast_evaluate_revised_experiments.py` | **5. Revised Evaluation** | Compares the original 18-feature models against the 21-feature humidity-augmented models, evaluates LODO performance across districts, and runs moving-block bootstrap sensitivity analyses across $7$-, $14$-, and $28$-day block lengths ($1,000$ replicates). |

---

## Experimental Design

### 1. Feature Sets (Strictly Leakage-Free)
All predictors are computed strictly from observations on day $T-1$ or earlier (plus known-in-advance cyclical calendar encodings for target date $T$)[cite: 3, 5]. District identifiers and same-day meteorological variables are strictly excluded.

* **Original Baseline (`original_18` — 18 features):**
  * **$\text{PM}_{2.5}$ Lags (5):** `pm25_lag_1d`, `pm25_lag_2d`, `pm25_lag_3d`, `pm25_lag_7d`, `pm25_lag_14d`[cite: 1, 3]
  * **$\text{PM}_{2.5}$ Prior Rolling Stats (4):** `pm25_roll_mean_prev_3d`, `pm25_roll_mean_prev_7d`, `pm25_roll_mean_prev_14d`, `pm25_roll_std_prev_7d`[cite: 1, 3]
  * **Wind Speed History (3):** `weather_0_wind_speed_mean_ms_lag_1d`, `weather_0_wind_speed_mean_ms_roll_mean_prev_3d`, `weather_0_wind_speed_mean_ms_roll_mean_prev_7d`[cite: 1, 3]
  * **Cyclical Calendar Encodings (6):** `day_of_year_sin`, `day_of_year_cos`, `day_of_week_sin`, `day_of_week_cos`, `month_sin`, `month_cos`[cite: 1, 3]
* **Humidity-Augmented (`corrected_plus_humidity` — 21 features):**
  * All 18 baseline features above plus 3 historical humidity features: `relative_humidity_lag_1d`, `relative_humidity_roll_mean_prev_3d`, and `relative_humidity_roll_mean_prev_7d`[cite: 1, 2].

### 2. Evaluation Regimes & Data Partitions
The prepared modeling dataset contains **$39,221$ district-day rows** across **13 districts** (`2017-09-28` to `2025-12-31`)[cite: 3, 4]:
* **Chronological Split:**
  * **Train:** Through `2023-07-09` ($27,443$ rows)
  * **Validation:** `2023-07-10` to `2024-10-04` ($5,889$ rows)
  * **Test:** `2024-10-05` to `2025-12-31` ($5,889$ rows)[cite: 3]
* **Random Splits (Reference Only):** Three independent random $70\% / 15\% / 15\%$ splits (`random_seed_42`, `random_seed_123`, `random_seed_2025`; $27,454$ train / $5,883$ validation / $5,884$ test rows) to quantify look-ahead optimism relative to chronological forecasting[cite: 3, 4, 5].
* **District Hold-Out Split:**
  * **Training & Validation Districts (10):** Faridpur, Gazipur, Gopalganj, Kishoreganj, Madaripur, Manikganj, Munshiganj, Rajbari, Shariatpur, and Tangail ($21,110$ train rows through `2023-07-09`; $4,530$ validation rows from `2023-07-10` to `2024-10-04`)[cite: 3].
  * **Held-Out Test Districts (3):** Dhaka, Narayanganj, and Narsingdi ($9,051$ test rows across all dates, plus matched future-window evaluation on `2024-10-05` to `2025-12-31`)[cite: 3, 4].
* **Leave-One-District-Out (LODO):** 13 spatial folds where each district is held out in turn and evaluated on the common future test window (`2024-10-05` to `2025-12-31`) after training on the remaining 12 districts (`<= 2023-07-09`)[cite: 1, 2].

### 3. Models & Hyperparameters
All learned models are CPU-based `scikit-learn` pipelines with median imputation (`SimpleImputer(strategy="median")`) fitted strictly on the training partition[cite: 2, 3]:
1. **Persistence (`persistence`):** Predicts $\hat{y}_t = \text{pm25\_lag\_1d}$ (no fitting)[cite: 3].
2. **Ridge Regression (`ridge`):** `SimpleImputer(median)` $\rightarrow$ `StandardScaler()` $\rightarrow$ `Ridge(alpha=1.0)`[cite: 2, 3].
3. **Random Forest (`random_forest`):** `SimpleImputer(median)` $\rightarrow$ `RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1)`[cite: 2, 3].
4. **Histogram Gradient Boosting (`hist_gbm`):** `SimpleImputer(median)` $\rightarrow$ `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False)`[cite: 2, 3].

---

## Execution Order

### Environment Setup
```bash
pip install -r requirements.txt
