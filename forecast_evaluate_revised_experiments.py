%%writefile /kaggle/working/evaluate_revised_experiments.py
# =====================================================================================
# EVALUATION OF REVISED PM2.5 EXPERIMENTS
# Reads predictions/models created by revised_experiments.py; does NOT retrain models.
# Produces CSV tables, figures, and one ZIP containing all CSVs and figures.
# =====================================================================================

import os, json, time, shutil, zipfile, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
T0 = time.time()
OUT_DIR = Path("/kaggle/working/revised_experiment_outputs")
MODEL_DIR = OUT_DIR / "models"
PRED_PATH = OUT_DIR / "all_revised_predictions.csv"
SPLIT_DIR = OUT_DIR / "splits"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
for p in (FIG_DIR, TAB_DIR): p.mkdir(parents=True, exist_ok=True)

DATE, DIST = "target_date", "district"
ACT, PRED = "actual_pm25_ug_m3", "predicted_pm25_ug_m3"
TARGET = "target_pm25_ug_m3"
BASE_FEATURES = [
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d", "pm25_lag_14d",
    "pm25_roll_mean_prev_3d", "pm25_roll_mean_prev_7d", "pm25_roll_mean_prev_14d",
    "pm25_roll_std_prev_7d",
    "weather_0_wind_speed_mean_ms_lag_1d",
    "weather_0_wind_speed_mean_ms_roll_mean_prev_3d",
    "weather_0_wind_speed_mean_ms_roll_mean_prev_7d",
    "day_of_year_sin", "day_of_year_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]
HUMIDITY_FEATURES = ["relative_humidity_lag_1d", "relative_humidity_roll_mean_prev_3d", "relative_humidity_roll_mean_prev_7d"]
FEATURE_SETS = {"original_18": BASE_FEATURES, "corrected_plus_humidity": BASE_FEATURES + HUMIDITY_FEATURES}
MODELS = ["persistence", "ridge", "random_forest", "hist_gbm"]
LEARNED = MODELS[1:]
MODEL_LABEL = {"persistence": "Persistence", "ridge": "Ridge", "random_forest": "Random Forest", "hist_gbm": "HistGBM"}
FEATURE_LABEL = {"original_18": "Original 18 features", "corrected_plus_humidity": "+ historical humidity"}
RANDOM_EXPS = ["random_seed_42", "random_seed_123", "random_seed_2025"]
BASE_EXPS = RANDOM_EXPS + ["chronological", "district_holdout"]
HELDOUT = ["Dhaka", "Narayanganj", "Narsingdi"]
TEST_START, TEST_END = pd.Timestamp("2024-10-05"), pd.Timestamp("2025-12-31")
BOOT_REPS = 1000
BLOCK_LENGTHS = [7, 14, 28]
SEED = 2048

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.22, "savefig.dpi": 300})
COLORS = {"persistence": "#777777", "ridge": "#E69F00", "random_forest": "#009E73", "hist_gbm": "#0072B2"}
FS_COLORS = {"original_18": "#777777", "corrected_plus_humidity": "#0072B2"}


def savefig(fig, stem):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def savecsv(df, name):
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)


def metrics(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    err = y - p
    sst = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "R2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else np.nan,
            "mean_residual_actual_minus_predicted": float(np.mean(err))}


def ensure_key_integrity(df, label):
    keys = [DATE, DIST]
    if df.duplicated(keys).any():
        raise RuntimeError(f"Duplicate date/district keys in {label}")
    if df[ACT].isna().any() or df[PRED].isna().any() or not np.isfinite(df[PRED].to_numpy(float)).all():
        raise RuntimeError(f"Missing/non-finite targets or predictions in {label}")


def prepare_date_groups(wide):
    """Precompute row positions for each date; all districts on a date stay together."""
    date_series = pd.to_datetime(wide[DATE], errors="coerce").dt.normalize()
    dates = pd.DatetimeIndex(date_series.unique()).sort_values()
    if len(dates) == 0:
        raise ValueError("No dates available for bootstrap")
    day_diffs = np.diff(dates.to_numpy(dtype="datetime64[D]")).astype(int)
    if len(day_diffs) and not np.all(day_diffs == 1):
        raise RuntimeError("Bootstrap input is missing calendar dates. Do not silently treat gaps as consecutive days.")
    date_values = date_series.to_numpy(dtype="datetime64[ns]")
    groups = [np.flatnonzero(date_values == np.datetime64(d, "ns")) for d in dates]
    return dates, groups


def block_row_indices_from_groups(dates, groups, block_days, rng):
    """Overlapping moving-block bootstrap over consecutive calendar dates."""
    n = len(dates)
    L = min(int(block_days), n)
    n_blocks = int(np.ceil(n / L))
    sampled = []
    for _ in range(n_blocks):
        start = int(rng.integers(0, n - L + 1))
        sampled.extend(range(start, start + L))
    sampled = sampled[:n]
    return np.concatenate([groups[j] for j in sampled])


def make_wide(exp, feature_sets, districts=None, date_min=None, date_max=None):
    sub = preds[(preds["experiment_id"] == exp) & (preds["partition"] == "test") &
                (preds["feature_set"].isin(feature_sets))].copy()
    sub[DATE] = pd.to_datetime(sub[DATE], errors="coerce").dt.normalize()
    if districts is not None: sub = sub[sub[DIST].isin(districts)]
    if date_min is not None: sub = sub[sub[DATE] >= pd.Timestamp(date_min)]
    if date_max is not None: sub = sub[sub[DATE] <= pd.Timestamp(date_max)]
    if sub.empty: raise RuntimeError(f"No prediction rows for {exp}, feature_sets={feature_sets}")
    # Start from one reference prediction table and merge all other predictions by key.
    first = sub[(sub["feature_set"] == feature_sets[0]) & (sub["model_name"] == MODELS[0])]
    if first.empty: raise RuntimeError(f"Missing reference rows for {exp}/{feature_sets[0]}/{MODELS[0]}")
    wide = first[[DATE, DIST, ACT]].copy().sort_values([DATE, DIST]).reset_index(drop=True)
    if wide.duplicated([DATE, DIST]).any(): raise RuntimeError("Duplicate reference keys")
    for fs in feature_sets:
        for m in MODELS:
            one = sub[(sub["feature_set"] == fs) & (sub["model_name"] == m)][[DATE, DIST, ACT, PRED]].copy()
            if one.empty: raise RuntimeError(f"Missing rows for {exp}/{fs}/{m}")
            if one.duplicated([DATE, DIST]).any(): raise RuntimeError(f"Duplicate rows for {exp}/{fs}/{m}")
            one = one.rename(columns={ACT: f"actual_check__{fs}__{m}", PRED: f"{fs}__{m}"})
            wide = wide.merge(one, on=[DATE, DIST], how="left", validate="one_to_one")
            acol = f"actual_check__{fs}__{m}"
            if wide[acol].isna().any() or not np.allclose(wide[ACT], wide[acol], rtol=0, atol=1e-8):
                raise RuntimeError(f"Actual target mismatch for {exp}/{fs}/{m}")
            wide = wide.drop(columns=[acol])
    wide = wide.sort_values([DATE, DIST]).reset_index(drop=True)
    return wide


def bootstrap_panel(wide, pred_columns, block_days, reps=BOOT_REPS, seed=SEED):
    rng = np.random.default_rng(seed)
    _, date_groups = prepare_date_groups(wide)
    y = wide[ACT].to_numpy(float)
    draws = {c: {"MAE": [], "RMSE": [], "R2": []} for c in pred_columns}
    for _ in range(reps):
        idx = block_row_indices_from_groups(np.arange(len(date_groups)), date_groups, block_days, rng)
        ys = y[idx]
        for c in pred_columns:
            ps = wide[c].to_numpy(float)[idx]
            mm = metrics(ys, ps)
            for k in ("MAE", "RMSE", "R2"): draws[c][k].append(mm[k])
    return draws


def ci(arr):
    a = np.asarray(arr, dtype=float)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ----------------------------- LOAD RESULTS -------------------------------------------
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing {PRED_PATH}. Run revised_experiments.py first.")
preds = pd.read_csv(PRED_PATH, low_memory=False)
required = {"experiment_id", "split_type", "feature_set", "model_name", "partition", DATE, DIST, ACT, PRED}
missing = sorted(required - set(preds.columns))
if missing: raise RuntimeError(f"Prediction table is missing columns: {missing}")
preds[DATE] = pd.to_datetime(preds[DATE], errors="coerce").dt.normalize()
preds[DIST] = preds[DIST].astype(str).str.strip()
preds[ACT] = pd.to_numeric(preds[ACT], errors="coerce")
preds[PRED] = pd.to_numeric(preds[PRED], errors="coerce")
if preds[[DATE, DIST, ACT, PRED]].isna().any().any() or not np.isfinite(preds[PRED].to_numpy(float)).all():
    raise RuntimeError("Prediction file contains missing dates, targets, or predictions")
if not {"original_18", "corrected_plus_humidity"}.issubset(set(preds["feature_set"])):
    raise RuntimeError("Expected original_18 and corrected_plus_humidity feature sets")
print(f"Loaded {len(preds):,} prediction rows from {PRED_PATH}")

# ----------------------------- TABLE 1: ALL METRICS ----------------------------------
metric_rows = []
for keys, g in preds.groupby(["experiment_id", "split_type", "feature_set", "model_name", "partition"], sort=False):
    exp, split, fs, model, part = keys
    metric_rows.append({"experiment_id": exp, "split_type": split, "feature_set": fs,
                        "model_name": model, "partition": part, **metrics(g[ACT], g[PRED]),
                        "n_districts": int(g[DIST].nunique()),
                        "date_min": str(g[DATE].min().date()), "date_max": str(g[DATE].max().date())})
metrics_all = pd.DataFrame(metric_rows)
# Skill is defined only relative to persistence within the exact same test rows / experiment / feature set.
base_rmse = metrics_all[metrics_all.model_name == "persistence"].set_index(
    ["experiment_id", "feature_set", "partition"])["RMSE"].to_dict()
metrics_all["RMSE_skill_vs_persistence"] = metrics_all.apply(
    lambda r: 1 - r["RMSE"] / base_rmse[(r["experiment_id"], r["feature_set"], r["partition"])], axis=1)
savecsv(metrics_all, "metrics_all_experiments")

# Random-split aggregate: mean and SD across seeds, shown separately from block-bootstrap CIs.
random_metrics = metrics_all[metrics_all.experiment_id.isin(RANDOM_EXPS) & (metrics_all.partition == "test")]
random_summary = random_metrics.groupby(["feature_set", "model_name"])[["MAE", "RMSE", "R2", "RMSE_skill_vs_persistence"]].agg(["mean", "std"])
random_summary.columns = [f"{a}_{b}" for a, b in random_summary.columns]
random_summary = random_summary.reset_index()
savecsv(random_summary, "random_seed_summary_mean_sd")

# ----------------------------- TABLE 2: OLD VS CORRECTED ------------------------------
comparison_rows = []
for exp in BASE_EXPS:
    for model in MODELS:
        old = preds[(preds.experiment_id == exp) & (preds.feature_set == "original_18") &
                    (preds.model_name == model) & (preds.partition == "test")].copy()
        new = preds[(preds.experiment_id == exp) & (preds.feature_set == "corrected_plus_humidity") &
                    (preds.model_name == model) & (preds.partition == "test")].copy()
        if old.empty or new.empty: continue
        old[DATE] = pd.to_datetime(old[DATE]).dt.normalize(); new[DATE] = pd.to_datetime(new[DATE]).dt.normalize()
        pair = old[[DATE, DIST, ACT, PRED]].merge(new[[DATE, DIST, ACT, PRED]], on=[DATE, DIST], suffixes=("_old", "_new"),
                                                  validate="one_to_one")
        if len(pair) != len(old) or len(pair) != len(new): raise RuntimeError(f"Unmatched rows for {exp}/{model}")
        if not np.allclose(pair[ACT + "_old"], pair[ACT + "_new"], rtol=0, atol=1e-8):
            raise RuntimeError(f"Targets differ for old/corrected: {exp}/{model}")
        mo, mn = metrics(pair[ACT + "_old"], pair[PRED + "_old"]), metrics(pair[ACT + "_new"], pair[PRED + "_new"])
        comparison_rows.append({"experiment_id": exp, "model_name": model, "n_matched_rows": len(pair),
                                "old_RMSE": mo["RMSE"], "corrected_RMSE": mn["RMSE"],
                                "delta_RMSE_corrected_minus_old": mn["RMSE"] - mo["RMSE"],
                                "delta_RMSE_pct": 100 * (mn["RMSE"] / mo["RMSE"] - 1) if mo["RMSE"] else np.nan,
                                "old_MAE": mo["MAE"], "corrected_MAE": mn["MAE"],
                                "delta_MAE_corrected_minus_old": mn["MAE"] - mo["MAE"],
                                "old_R2": mo["R2"], "corrected_R2": mn["R2"]})
comparison = pd.DataFrame(comparison_rows)
savecsv(comparison, "original_vs_humidity_corrected_matched_tests")

# Random test rows scored only within the future chronological-test window, to compare like-for-like dates.
same_window_rows = []
for exp in RANDOM_EXPS:
    for fs in FEATURE_SETS:
        for model in MODELS:
            g = preds[(preds.experiment_id == exp) & (preds.feature_set == fs) &
                      (preds.model_name == model) & (preds.partition == "test")].copy()
            g = g[(g[DATE] >= TEST_START) & (g[DATE] <= TEST_END)]
            if g.empty: continue
            same_window_rows.append({"experiment_id": exp, "feature_set": fs, "model_name": model,
                                     "window": "2024-10-05 to 2025-12-31", **metrics(g[ACT], g[PRED])})
same_window = pd.DataFrame(same_window_rows)
savecsv(same_window, "random_test_predictions_restricted_to_chronological_window")

# ----------------------------- TABLE 3: LODO -----------------------------------------
lodo = metrics_all[(metrics_all.split_type == "leave_one_district_out") & (metrics_all.partition == "test")].copy()
if not lodo.empty:
    lodo["heldout_district"] = lodo["experiment_id"].str.replace(r"^lodo_", "", regex=True).str.replace("_", " ")
    savecsv(lodo, "lodo_metrics_by_district")
    lodo_summary = lodo.groupby(["feature_set", "model_name"])[["MAE", "RMSE", "R2", "RMSE_skill_vs_persistence"]].agg(["mean", "std", "min", "max"])
    lodo_summary.columns = [f"{a}_{b}" for a, b in lodo_summary.columns]
    lodo_summary = lodo_summary.reset_index()
    savecsv(lodo_summary, "lodo_summary_across_districts")
else:
    raise RuntimeError("No LODO test predictions found in revised predictions table")

# ----------------------------- BOOTSTRAP: CHRONOLOGICAL -------------------------------
bootstrap_rows = []
for block_len in BLOCK_LENGTHS:
    wide = make_wide("chronological", ["original_18", "corrected_plus_humidity"])
    pred_cols = [f"{fs}__{m}" for fs in FEATURE_SETS for m in MODELS]
    draws = bootstrap_panel(wide, pred_cols, block_len, reps=BOOT_REPS, seed=SEED + block_len)
    for fs in FEATURE_SETS:
        for model in MODELS:
            c = f"{fs}__{model}"
            pt = metrics(wide[ACT], wide[c])
            lo_rmse, hi_rmse = ci(draws[c]["RMSE"])
            lo_mae, hi_mae = ci(draws[c]["MAE"])
            skill_arr = 1 - np.asarray(draws[c]["RMSE"]) / np.asarray(draws[f"{fs}__persistence"]["RMSE"])
            skill_pt = 1 - pt["RMSE"] / metrics(wide[ACT], wide[f"{fs}__persistence"])["RMSE"]
            lo_skill, hi_skill = ci(skill_arr)
            bootstrap_rows.append({"analysis": "chronological", "block_days": block_len, "feature_set": fs,
                                   "model_name": model, "n_rows": len(wide), "n_dates": wide[DATE].nunique(),
                                   "MAE": pt["MAE"], "MAE_ci_lo": lo_mae, "MAE_ci_hi": hi_mae,
                                   "RMSE": pt["RMSE"], "RMSE_ci_lo": lo_rmse, "RMSE_ci_hi": hi_rmse,
                                   "R2": pt["R2"], "RMSE_skill_vs_persistence": skill_pt,
                                   "skill_ci_lo": lo_skill, "skill_ci_hi": hi_skill})
        # Paired effect of adding humidity: same dates/rows/resamples for original vs corrected.
        for model in MODELS:
            old = f"original_18__{model}"; new = f"corrected_plus_humidity__{model}"
            diff = np.asarray(draws[new]["RMSE"]) - np.asarray(draws[old]["RMSE"])
            old_pt, new_pt = metrics(wide[ACT], wide[old]), metrics(wide[ACT], wide[new])
            dlo, dhi = ci(diff)
            bootstrap_rows.append({"analysis": "chronological_paired_feature_effect", "block_days": block_len,
                                   "feature_set": "corrected_minus_original", "model_name": model,
                                   "n_rows": len(wide), "n_dates": wide[DATE].nunique(),
                                   "RMSE_original": old_pt["RMSE"], "RMSE_corrected": new_pt["RMSE"],
                                   "delta_RMSE_corrected_minus_original": new_pt["RMSE"] - old_pt["RMSE"],
                                   "delta_RMSE_ci_lo": dlo, "delta_RMSE_ci_hi": dhi})
bootstrap_df = pd.DataFrame(bootstrap_rows)
savecsv(bootstrap_df, "moving_block_bootstrap_chronological_sensitivity_7_14_28d")

# ----------------------------- BOOTSTRAP: SEEN VS UNSEEN ------------------------------
matched_boot_rows = []
for fs in FEATURE_SETS:
    for model in MODELS:
        seen = preds[(preds.experiment_id == "chronological") & (preds.feature_set == fs) &
                     (preds.model_name == model) & (preds.partition == "test") &
                     (preds[DIST].isin(HELDOUT))].copy()
        unseen = preds[(preds.experiment_id == "district_holdout") & (preds.feature_set == fs) &
                       (preds.model_name == model) & (preds.partition == "test") &
                       (preds[DIST].isin(HELDOUT))].copy()
        seen = seen[(seen[DATE] >= TEST_START) & (seen[DATE] <= TEST_END)]
        unseen = unseen[(unseen[DATE] >= TEST_START) & (unseen[DATE] <= TEST_END)]
        seen = seen[[DATE, DIST, ACT, PRED]].rename(columns={PRED: "seen_pred"})
        unseen = unseen[[DATE, DIST, ACT, PRED]].rename(columns={PRED: "unseen_pred", ACT: "actual_unseen"})
        wide = seen.merge(unseen, on=[DATE, DIST], how="inner", validate="one_to_one")
        if wide.empty or len(wide) != len(seen) or len(wide) != len(unseen):
            raise RuntimeError(f"Matched seen/unseen rows not aligned for {fs}/{model}")
        if not np.allclose(wide[ACT], wide["actual_unseen"], rtol=0, atol=1e-8):
            raise RuntimeError("Seen/unseen target values differ")
        wide = wide.drop(columns=["actual_unseen"]).sort_values([DATE, DIST]).reset_index(drop=True)
        dates_for_boot, groups_for_boot = prepare_date_groups(wide)
        for block_len in BLOCK_LENGTHS:
            rng = np.random.default_rng(SEED + block_len + len(matched_boot_rows))
            diff_rmse, seen_rmse, unseen_rmse = [], [], []
            for _ in range(BOOT_REPS):
                idx = block_row_indices_from_groups(dates_for_boot, groups_for_boot, block_len, rng)
                y = wide[ACT].to_numpy(float)[idx]
                ps = wide["seen_pred"].to_numpy(float)[idx]
                pu = wide["unseen_pred"].to_numpy(float)[idx]
                rms = float(np.sqrt(np.mean((y - ps) ** 2)))
                rmu = float(np.sqrt(np.mean((y - pu) ** 2)))
                seen_rmse.append(rms); unseen_rmse.append(rmu); diff_rmse.append(rmu - rms)
            point_seen = metrics(wide[ACT], wide["seen_pred"])["RMSE"]
            point_unseen = metrics(wide[ACT], wide["unseen_pred"])["RMSE"]
            dlo, dhi = ci(diff_rmse); slo, shi = ci(seen_rmse); ulo, uhi = ci(unseen_rmse)
            matched_boot_rows.append({"feature_set": fs, "model_name": model, "block_days": block_len,
                                      "n_rows": len(wide), "n_dates": wide[DATE].nunique(),
                                      "RMSE_seen": point_seen, "RMSE_seen_ci_lo": slo, "RMSE_seen_ci_hi": shi,
                                      "RMSE_unseen": point_unseen, "RMSE_unseen_ci_lo": ulo, "RMSE_unseen_ci_hi": uhi,
                                      "delta_RMSE_unseen_minus_seen": point_unseen - point_seen,
                                      "delta_ci_lo": dlo, "delta_ci_hi": dhi})
matched_boot_df = pd.DataFrame(matched_boot_rows)
savecsv(matched_boot_df, "moving_block_bootstrap_seen_vs_unseen_sensitivity_7_14_28d")

# ----------------------------- PERMUTATION IMPORTANCE -------------------------------
importance_rows = []
chrono_split_path = SPLIT_DIR / "chronological__test.csv"
chrono_model_path = MODEL_DIR / "chronological__corrected_plus_humidity__hist_gbm.joblib"
if chrono_split_path.exists() and chrono_model_path.exists():
    test_frame = pd.read_csv(chrono_split_path, low_memory=False)
    model = joblib.load(chrono_model_path)
    features = FEATURE_SETS["corrected_plus_humidity"]
    absent = [f for f in features if f not in test_frame.columns]
    if absent:
        print("Skipping permutation importance; missing test features:", absent)
    else:
        X = test_frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(test_frame[TARGET], errors="coerce").to_numpy(float)
        baseline = float(np.sqrt(np.mean((y - model.predict(X)) ** 2)))
        rng = np.random.default_rng(SEED)
        for feature in features:
            deltas = []
            for _ in range(5):
                perm = rng.permutation(len(X))
                xp = X.copy()
                xp[feature] = X[feature].to_numpy()[perm]
                deltas.append(float(np.sqrt(np.mean((y - model.predict(xp)) ** 2)) - baseline))
            importance_rows.append({"feature": feature, "group": "humidity" if feature in HUMIDITY_FEATURES else
                                    ("PM2.5 history" if feature.startswith("pm25_") else
                                     ("wind history" if feature.startswith("weather_") else "calendar")),
                                    "baseline_rmse": baseline, "delta_rmse_mean": float(np.mean(deltas)),
                                    "delta_rmse_sd": float(np.std(deltas, ddof=1)), "n_repeats": 5})
        importance = pd.DataFrame(importance_rows).sort_values("delta_rmse_mean", ascending=False)
        savecsv(importance, "permutation_importance_corrected_chronological_histgbm")
else:
    importance = pd.DataFrame()
    print("Permutation importance skipped because the corrected chronological HistGBM model/test split is missing.")

# ----------------------------- FIGURES -----------------------------------------------
# Figure 1: chronological RMSE old vs corrected, same rows.
chron_m = metrics_all[(metrics_all.experiment_id == "chronological") & (metrics_all.partition == "test")]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(MODELS)); width = 0.36
for j, fs in enumerate(FEATURE_SETS):
    vals = [chron_m[(chron_m.feature_set == fs) & (chron_m.model_name == m)]["RMSE"].iloc[0] for m in MODELS]
    ax.bar(x + (j - 0.5) * width, vals, width, label=FEATURE_LABEL[fs], color=FS_COLORS[fs])
ax.set_xticks(x); ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
ax.set_ylabel("RMSE (µg/m³)"); ax.set_title("Chronological test performance: original vs humidity-augmented features")
ax.legend(frameon=False); savefig(fig, "fig01_chronological_original_vs_corrected_rmse")

# Figure 2: leave-one-district-out RMSE.
fig, ax = plt.subplots(figsize=(12, 5.5))
if not lodo.empty:
    districts_order = sorted(lodo["heldout_district"].unique())
    labels = [x.replace("_", " ") for x in districts_order]
    offsets = {"persistence": -0.30, "ridge": -0.10, "random_forest": 0.10, "hist_gbm": 0.30}
    colorset = {"original_18": "#999999", "corrected_plus_humidity": "#0072B2"}
    for fs in FEATURE_SETS:
        for m in MODELS:
            g = lodo[(lodo.feature_set == fs) & (lodo.model_name == m)].set_index("heldout_district").reindex(districts_order)
            if g["RMSE"].notna().any():
                xx = np.arange(len(districts_order)) + offsets[m] + (-0.035 if fs == "original_18" else 0.035)
                ax.scatter(xx, g["RMSE"], s=25, marker="o" if fs == "original_18" else "s",
                           color=COLORS[m], alpha=0.72, label=f"{MODEL_LABEL[m]} — {FEATURE_LABEL[fs]}")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Future-period RMSE (µg/m³)"); ax.set_title("Leave-one-district-out performance on the common future test period")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    savefig(fig, "fig02_lodo_rmse_by_heldout_district")
else:
    plt.close(fig)

# Figure 3: block length sensitivity on chronological RMSE.
fig, ax = plt.subplots(figsize=(8.5, 4.5))
boot_metric = bootstrap_df[bootstrap_df.analysis == "chronological"]
for fs in FEATURE_SETS:
    for m in ("persistence", "random_forest", "hist_gbm"):
        g = boot_metric[(boot_metric.feature_set == fs) & (boot_metric.model_name == m)].sort_values("block_days")
        if g.empty: continue
        mid = g["RMSE"].to_numpy(float); lo = g["RMSE_ci_lo"].to_numpy(float); hi = g["RMSE_ci_hi"].to_numpy(float)
        lbl = f"{MODEL_LABEL[m]} ({'corrected' if fs == 'corrected_plus_humidity' else 'original'})"
        ax.errorbar(g["block_days"], mid, yerr=[mid - lo, hi - mid], marker="o", capsize=3, label=lbl)
ax.set_xticks(BLOCK_LENGTHS); ax.set_xlabel("Moving-block length (calendar days)"); ax.set_ylabel("Chronological RMSE (µg/m³)")
ax.set_title("Uncertainty sensitivity to bootstrap block length"); ax.legend(frameon=False, fontsize=7)
savefig(fig, "fig03_bootstrap_block_length_sensitivity")

# Figure 4: permutation importance.
if not importance.empty:
    top = importance.head(15).sort_values("delta_rmse_mean")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(top["feature"], top["delta_rmse_mean"], xerr=top["delta_rmse_sd"], color="#0072B2", alpha=0.85, capsize=2)
    ax.set_xlabel("Increase in RMSE after permutation (µg/m³)")
    ax.set_title("Permutation importance: corrected chronological HistGBM")
    savefig(fig, "fig04_corrected_histgbm_permutation_importance")

# Figure 5: corrected-minus-original paired RMSE difference.
if not comparison.empty:
    c = comparison[comparison.experiment_id == "chronological"].set_index("model_name").reindex(MODELS)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    vals = c["delta_RMSE_corrected_minus_old"].to_numpy(float)
    ax.bar([MODEL_LABEL[m] for m in MODELS], vals, color=[COLORS[m] for m in MODELS])
    ax.axhline(0, color="black", lw=0.9)
    ax.set_ylabel("Corrected − original RMSE (µg/m³)")
    ax.set_title("Matched chronological test: effect of adding historical humidity")
    savefig(fig, "fig05_humidity_effect_chronological")

# ----------------------------- SUMMARY AND ZIP ---------------------------------------
summary = {
    "prediction_rows": int(len(preds)),
    "experiments": sorted(preds["experiment_id"].unique().tolist()),
    "feature_sets": sorted(preds["feature_set"].unique().tolist()),
    "models": MODELS,
    "bootstrap_replicates": BOOT_REPS,
    "bootstrap_block_days": BLOCK_LENGTHS,
    "bootstrap_method": "Overlapping moving blocks of consecutive calendar dates; all districts from a date resampled together.",
    "target_caveat": "CAMS/ERA5-derived estimated PM2.5; not direct ground-monitor measurements.",
    "output_tables": sorted([p.name for p in TAB_DIR.glob("*.csv")]),
    "output_figures": sorted([p.name for p in FIG_DIR.glob("*.png")]),
    "important_caveat": "Metrics evaluate agreement with the model-derived target. They do not validate concentration accuracy against ground observations."
}
with open(OUT_DIR / "revised_evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

# ZIP all CSV tables and figure files (PNG + PDF), plus useful JSON/log metadata.
zip_path = Path("/kaggle/working/revised_pm25_results_csv_figures.zip")
if zip_path.exists(): zip_path.unlink()
include_suffixes = {".csv", ".png", ".pdf", ".json", ".log"}
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for p in OUT_DIR.rglob("*"):
        if not p.is_file() or p == zip_path: continue
        if p.suffix.lower() in include_suffixes:
            zf.write(p, arcname=str(p.relative_to(OUT_DIR)))
print("\nCore metrics:")
print(metrics_all[(metrics_all.partition == "test") & (metrics_all.experiment_id == "chronological")][
    ["feature_set", "model_name", "n", "MAE", "RMSE", "R2", "RMSE_skill_vs_persistence"]
].round(3).to_string(index=False))
print("\nLODO average metrics:")
print(lodo_summary.round(3).to_string(index=False))
print(f"\nTables: {TAB_DIR}")
print(f"Figures: {FIG_DIR}")
print(f"Downloadable archive created: {zip_path}")
print(f"Evaluation completed in {(time.time() - T0)/60:.1f} minutes.")



Cell 2:

!python /kaggle/working/evaluate_revised_experiments.py
from pathlib import Path
from IPython.display import FileLink, display
zip_file = Path('/kaggle/working/revised_pm25_results_csv_figures.zip')
assert zip_file.exists(), 'ZIP was not created; inspect the evaluation errors above.'
display(FileLink(str(zip_file)))