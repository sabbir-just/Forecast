# =====================================================================================
# EVALUATION CELL
# Temporal and spatial generalization of lightweight ML models for next-day
# ESTIMATED PM2.5 forecasting (CAMS/ERA5-derived estimates), Dhaka Division.
#
# Loads saved artifacts only. No model is re-trained here.
#   1. Integrity checks of training artifacts
#   2. Metrics (MAE, RMSE, R2, nRMSE, skill vs persistence), seed aggregation
#   3. Block-bootstrap CIs (14-day blocks) and paired comparisons
#   4. RQ1: random vs chronological (incl. same-window decomposition)
#   5. RQ2: unseen districts (incl. matched seen-vs-unseen comparison)
#   6. RQ3: permutation importance (features + groups) and ranking stability
#   7. Publication-quality figures (PNG 300 dpi + PDF) and tables (CSV/LaTeX)
# =====================================================================================

import os, json, time, shutil, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from IPython.display import display

warnings.filterwarnings("ignore")
T_START = time.time()

# -------------------------------------------------------------------------------------
# 0. SETTINGS
# -------------------------------------------------------------------------------------
PREP_DIR = Path("/kaggle/working/pretraining_outputs")
TRAIN_DIR = Path("/kaggle/working/training_outputs")
EVAL_DIR = Path("/kaggle/working/evaluation_outputs")
FIG_DIR = EVAL_DIR / "figures"
TAB_DIR = EVAL_DIR / "tables"
for d in (EVAL_DIR, FIG_DIR, TAB_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 2000            # block-bootstrap replicates
BLOCK_DAYS = 14          # bootstrap block length (days) to respect autocorrelation
N_PERM_REPEATS = 10      # permutation repeats (reduce to 5 for a faster run)
REUSE_IMPORTANCE = True  # reuse tables/permutation_importance_long.csv if present

TARGET, DATE, DIST = "target_pm25_ug_m3", "target_date", "district"
ACT, PRED = "actual_pm25_ug_m3", "predicted_pm25_ug_m3"

EXPERIMENTS = ["random_seed_42", "random_seed_123", "random_seed_2025",
               "chronological", "district_holdout"]
RANDOM_EXPS = EXPERIMENTS[:3]
MODELS = ["persistence", "ridge", "random_forest", "hist_gbm"]
LEARNED = MODELS[1:]

MODEL_LABEL = {"persistence": "Persistence", "ridge": "Ridge",
               "random_forest": "Random Forest", "hist_gbm": "HistGBM"}
EXP_LABEL = {"random_seed_42": "Random (seed 42)", "random_seed_123": "Random (seed 123)",
             "random_seed_2025": "Random (seed 2025)", "chronological": "Chronological",
             "district_holdout": "District hold-out"}
COL = {"persistence": "#8c8c8c", "ridge": "#E69F00",
       "random_forest": "#009E73", "hist_gbm": "#0072B2"}

HELDOUT = ["Dhaka", "Narayanganj", "Narsingdi"]
CS, CE = pd.Timestamp("2024-10-05"), pd.Timestamp("2025-12-31")
PERIODS = {
    "Train-era dates\n2017-09-28 to 2023-07-09": (pd.Timestamp("2017-09-28"), pd.Timestamp("2023-07-09")),
    "Validation-era dates\n2023-07-10 to 2024-10-04": (pd.Timestamp("2023-07-10"), pd.Timestamp("2024-10-04")),
    "Test-era dates\n2024-10-05 to 2025-12-31": (CS, CE),
}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "figure.dpi": 110, "savefig.dpi": 300,
})


def savefig(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def savetab(df, name, index=False):
    df.to_csv(TAB_DIR / f"{name}.csv", index=index)


def parse_dates(s):
    try:
        out = pd.to_datetime(s, format="mixed")
    except (TypeError, ValueError):
        out = pd.to_datetime(s)
    return out.dt.normalize()


def short_name(f):
    return (f.replace("weather_0_wind_speed_mean_ms", "wind")
             .replace("roll_mean_prev", "rollmean").replace("roll_std_prev", "rollstd"))


# -------------------------------------------------------------------------------------
# 1. LOAD AND VERIFY TRAINING ARTIFACTS
# -------------------------------------------------------------------------------------
manifest = pd.read_csv(TRAIN_DIR / "completed_experiments_manifest.csv")
if len(manifest) != 20 or set(manifest["experiment_id"]) != set(EXPERIMENTS) \
        or set(manifest["model_name"]) != set(MODELS):
    raise RuntimeError("Manifest does not contain the expected 5 experiments x 4 models.")

preds = pd.read_csv(TRAIN_DIR / "all_predictions.csv")
preds[DATE] = parse_dates(preds[DATE])
expected_rows = int((manifest["n_validation"] + manifest["n_test"]).sum())
checks = {
    "manifest_rows": int(len(manifest)),
    "prediction_rows": int(len(preds)),
    "prediction_rows_expected": expected_rows,
    "prediction_nan": int(preds[PRED].isna().sum()),
    "prediction_duplicates": int(preds.duplicated(
        ["experiment_id", "model_name", "partition", DIST, DATE]).sum()),
}
if checks["prediction_rows"] != expected_rows or checks["prediction_nan"] or \
        checks["prediction_duplicates"]:
    raise RuntimeError(f"Prediction table failed integrity checks: {checks}")

with open(TRAIN_DIR / "feature_audit.json") as f:
    FEATURES = json.load(f)["eligible_features"]
print(f"Loaded {len(preds):,} predictions; {len(FEATURES)} features; integrity checks passed.")


# -------------------------------------------------------------------------------------
# 2. METRICS
# -------------------------------------------------------------------------------------
def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    e = y - p
    sse = float(np.sum(e * e))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(sse / len(y))),
            "R2": 1 - sse / sst if sst > 0 else np.nan,
            "mean_residual": float(e.mean()),  # actual - predicted (>0: under-prediction)
            "y_mean": float(y.mean()), "y_std": float(y.std(ddof=1))}


rows = []
for (e, m, part), g in preds.groupby(["experiment_id", "model_name", "partition"]):
    rows.append({"experiment_id": e, "model_name": m, "partition": part,
                 **metrics(g[ACT], g[PRED]), "n_negative_predictions": int((g[PRED] < 0).sum())})
metrics_long = pd.DataFrame(rows)
pers_rmse = metrics_long[metrics_long.model_name == "persistence"].set_index(
    ["experiment_id", "partition"])["RMSE"].to_dict()
metrics_long["skill_rmse_vs_persistence"] = metrics_long.apply(
    lambda r: 1 - r["RMSE"] / pers_rmse[(r["experiment_id"], r["partition"])], axis=1)
metrics_long["nRMSE"] = metrics_long["RMSE"] / metrics_long["y_mean"]

# Cross-check against the training manifest.
chk = metrics_long[metrics_long.partition == "test"].merge(
    manifest[["experiment_id", "model_name", "test_rmse", "test_mae", "test_r2"]],
    on=["experiment_id", "model_name"])
checks["max_abs_diff_vs_manifest_test_rmse"] = float((chk["RMSE"] - chk["test_rmse"]).abs().max())
checks["max_abs_diff_vs_manifest_test_r2"] = float((chk["R2"] - chk["test_r2"]).abs().max())
if checks["max_abs_diff_vs_manifest_test_rmse"] > 1e-6:
    raise RuntimeError("Recomputed test RMSE disagrees with the training manifest.")
savetab(metrics_long, "T1_metrics_all_experiments")

TESTROW = {(r.experiment_id, r.model_name): r
           for r in metrics_long[metrics_long.partition == "test"].itertuples()}

# Seed aggregation for the random-split reference.
rnd = metrics_long[metrics_long.experiment_id.isin(RANDOM_EXPS)]
rand_agg = rnd.groupby(["model_name", "partition"])[
    ["MAE", "RMSE", "R2", "nRMSE", "skill_rmse_vs_persistence"]].agg(["mean", "std"])
rand_agg.columns = [f"{a}_{b}" for a, b in rand_agg.columns]
rand_agg = rand_agg.reset_index()
savetab(rand_agg, "T2_random_split_seed_aggregate")

# Model selection by VALIDATION RMSE only (never the test set).
val_ml = metrics_long[(metrics_long.partition == "validation") & metrics_long.model_name.isin(LEARNED)]
SELECTED = val_ml.loc[val_ml.groupby("experiment_id")["RMSE"].idxmin()].set_index(
    "experiment_id")["model_name"].to_dict()
print("Model selected by validation RMSE per experiment:", SELECTED)

# -------------------------------------------------------------------------------------
# 3. ALIGNED PREDICTIONS AND BLOCK BOOTSTRAP
# -------------------------------------------------------------------------------------
ALIGN = {}
for exp in EXPERIMENTS:
    for part in ("validation", "test"):
        sub = preds[(preds.experiment_id == exp) & (preds.partition == part)]
        ref, out = None, {}
        for m in MODELS:
            s = sub[sub.model_name == m].reset_index(drop=True)
            if ref is None:
                ref = s[[DATE, DIST, ACT]].copy()
            elif not (s[DATE].equals(ref[DATE]) and s[DIST].equals(ref[DIST])
                      and np.allclose(s[ACT], ref[ACT], rtol=0, atol=1e-9)):
                raise RuntimeError(f"Row misalignment between models in {exp}/{part}")
            out[m] = s[PRED].to_numpy(float)
        ALIGN[(exp, part)] = (ref, out)


def block_ids(dates, block_days=BLOCK_DAYS):
    ids = ((dates - dates.min()).dt.days // block_days).to_numpy()
    _, inv = np.unique(ids, return_inverse=True)
    return inv


def make_W(B, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(N_BOOT, B))
    return np.stack([np.bincount(r, minlength=B) for r in idx]).astype(float)


def block_sums(y, p, bid, B):
    e = y - p
    return {"n": np.bincount(bid, minlength=B).astype(float),
            "sy": np.bincount(bid, weights=y, minlength=B),
            "syy": np.bincount(bid, weights=y * y, minlength=B),
            "sae": np.bincount(bid, weights=np.abs(e), minlength=B),
            "sse": np.bincount(bid, weights=e * e, minlength=B)}


def stats_from_W(S, W):
    n, sy, syy = W @ S["n"], W @ S["sy"], W @ S["syy"]
    sae, sse = W @ S["sae"], W @ S["sse"]
    return {"MAE": sae / n, "RMSE": np.sqrt(sse / n), "R2": 1 - sse / (syy - sy ** 2 / n)}


def get_ci(a):
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


BOOT = {}
for i, exp in enumerate(EXPERIMENTS):
    ref, pr = ALIGN[(exp, "test")]
    y = ref[ACT].to_numpy(float)
    bid = block_ids(ref[DATE])
    B = int(bid.max()) + 1
    W = make_W(B, SEED + i)
    BOOT[exp] = {"stats": {m: stats_from_W(block_sums(y, pr[m], bid, B), W) for m in MODELS},
                 "n_blocks": B}


def point_ci(exp, m, key):
    st = BOOT[exp]["stats"]
    r = TESTROW[(exp, m)]
    if key == "skill":
        arr = 1 - st[m]["RMSE"] / st["persistence"]["RMSE"]
        pt = r.skill_rmse_vs_persistence
    else:
        arr, pt = st[m][key], getattr(r, key)
    lo, hi = get_ci(arr)
    return float(pt), lo, hi


# T3: test metrics with 95% block-bootstrap CIs
t3 = []
for exp in EXPERIMENTS:
    for m in MODELS:
        rec = {"experiment_id": exp, "model_name": m, "n_test": TESTROW[(exp, m)].n,
               "n_blocks": BOOT[exp]["n_blocks"]}
        for key in ("MAE", "RMSE", "R2", "skill"):
            if key == "skill" and m == "persistence":
                continue
            pt, lo, hi = point_ci(exp, m, key)
            rec.update({key: pt, f"{key}_ci_lo": lo, f"{key}_ci_hi": hi})
        t3.append(rec)
t3 = pd.DataFrame(t3)
savetab(t3, "T3_test_metrics_with_block_bootstrap_ci")

# T4: paired comparisons against persistence and among learned models
t4 = []
for exp in EXPERIMENTS:
    st = BOOT[exp]["stats"]
    for m in LEARNED:
        d_rmse = st[m]["RMSE"] - st["persistence"]["RMSE"]
        d_mae = st[m]["MAE"] - st["persistence"]["MAE"]
        sk = 1 - st[m]["RMSE"] / st["persistence"]["RMSE"]
        t4.append({"experiment_id": exp, "comparison": f"{m} vs persistence",
                   "dRMSE": TESTROW[(exp, m)].RMSE - TESTROW[(exp, "persistence")].RMSE,
                   "dRMSE_ci_lo": get_ci(d_rmse)[0], "dRMSE_ci_hi": get_ci(d_rmse)[1],
                   "dMAE": TESTROW[(exp, m)].MAE - TESTROW[(exp, "persistence")].MAE,
                   "dMAE_ci_lo": get_ci(d_mae)[0], "dMAE_ci_hi": get_ci(d_mae)[1],
                   "skill_rmse": TESTROW[(exp, m)].skill_rmse_vs_persistence,
                   "skill_ci_lo": get_ci(sk)[0], "skill_ci_hi": get_ci(sk)[1],
                   "p_model_better": float(np.mean(d_rmse < 0))})
    for a, b in combinations(LEARNED, 2):
        d = st[a]["RMSE"] - st[b]["RMSE"]
        t4.append({"experiment_id": exp, "comparison": f"{a} vs {b}",
                   "dRMSE": TESTROW[(exp, a)].RMSE - TESTROW[(exp, b)].RMSE,
                   "dRMSE_ci_lo": get_ci(d)[0], "dRMSE_ci_hi": get_ci(d)[1],
                   "p_model_better": float(np.mean(d < 0))})
t4 = pd.DataFrame(t4)
savetab(t4, "T4_paired_comparisons")

# -------------------------------------------------------------------------------------
# 4. RQ1: RANDOM vs CHRONOLOGICAL
# -------------------------------------------------------------------------------------
C_ALL = "Random split, all test dates"
C_WIN = "Random split, chronological-test window only"
C_CHR = "Chronological (future dates)"
KEYS = ["MAE", "RMSE", "R2", "skill", "nRMSE", "y_mean"]
rq1 = []
for m in MODELS:
    for label, window in ((C_ALL, False), (C_WIN, True)):
        recs = []
        for exp in RANDOM_EXPS:
            ref, pr = ALIGN[(exp, "test")]
            mask = np.ones(len(ref), bool) if not window else \
                ((ref[DATE] >= CS) & (ref[DATE] <= CE)).to_numpy()
            y = ref[ACT].to_numpy(float)[mask]
            mt, mp = metrics(y, pr[m][mask]), metrics(y, pr["persistence"][mask])
            mt["skill"] = 1 - mt["RMSE"] / mp["RMSE"]
            mt["nRMSE"] = mt["RMSE"] / mt["y_mean"]
            recs.append(mt)
        df = pd.DataFrame(recs)
        rec = {"model": m, "condition": label, "n_per_seed": float(df["n"].mean())}
        for k in KEYS:
            rec[f"{k}_mean"], rec[f"{k}_sd"] = float(df[k].mean()), float(df[k].std(ddof=1))
        rq1.append(rec)
    r = TESTROW[("chronological", m)]
    rec = {"model": m, "condition": C_CHR, "n_per_seed": r.n}
    for k, v in (("MAE", r.MAE), ("RMSE", r.RMSE), ("R2", r.R2), ("nRMSE", r.nRMSE),
                 ("y_mean", r.y_mean), ("skill", r.skill_rmse_vs_persistence)):
        rec[f"{k}_mean"], rec[f"{k}_sd"] = float(v), np.nan
    if m != "persistence":
        rec["RMSE_lo"], rec["RMSE_hi"] = point_ci("chronological", m, "RMSE")[1:]
        rec["skill_lo"], rec["skill_hi"] = point_ci("chronological", m, "skill")[1:]
    else:
        rec["RMSE_lo"], rec["RMSE_hi"] = point_ci("chronological", m, "RMSE")[1:]
    rq1.append(rec)
rq1_df = pd.DataFrame(rq1)
savetab(rq1_df, "T5_rq1_random_vs_chronological")

gap = []
for m in MODELS:
    a = rq1_df[(rq1_df.model == m) & (rq1_df.condition == C_ALL)].iloc[0]
    b = rq1_df[(rq1_df.model == m) & (rq1_df.condition == C_WIN)].iloc[0]
    c = rq1_df[(rq1_df.model == m) & (rq1_df.condition == C_CHR)].iloc[0]
    gap.append({"model": m, "RMSE_random_all": a.RMSE_mean, "RMSE_random_same_window": b.RMSE_mean,
                "RMSE_chronological": c.RMSE_mean,
                "dRMSE_chron_minus_random_all": c.RMSE_mean - a.RMSE_mean,
                "rel_change_pct_vs_random_all": 100 * (c.RMSE_mean / a.RMSE_mean - 1),
                "dRMSE_chron_minus_random_same_window": c.RMSE_mean - b.RMSE_mean,
                "rel_change_pct_vs_random_same_window": 100 * (c.RMSE_mean / b.RMSE_mean - 1),
                "R2_random_all": a.R2_mean, "R2_chronological": c.R2_mean,
                "skill_random_all": a.skill_mean, "skill_random_same_window": b.skill_mean,
                "skill_chronological": c.skill_mean})
rq1_gap = pd.DataFrame(gap)
savetab(rq1_gap, "T5b_rq1_gap_summary")

# -------------------------------------------------------------------------------------
# 5. RQ2: SPATIAL GENERALIZATION
# -------------------------------------------------------------------------------------
def to_full(exp, part="test"):
    ref, pr = ALIGN[(exp, part)]
    full = ref.copy()
    for m in MODELS:
        full[m] = pr[m]
    return full


chron_full, dh_full = to_full("chronological"), to_full("district_holdout")

# 5a. Held-out test: per district and per period slice
rq2_district, rq2_period = [], []
for dname, g in dh_full.groupby(DIST):
    for m in MODELS:
        rq2_district.append({"district": dname, "model": m, **metrics(g[ACT], g[m])})
rq2_district = pd.DataFrame(rq2_district)
savetab(rq2_district, "T6a_rq2_heldout_by_district_all_dates")

for plabel, (a, b) in PERIODS.items():
    g = dh_full[(dh_full[DATE] >= a) & (dh_full[DATE] <= b)]
    pr_rmse = metrics(g[ACT], g["persistence"])["RMSE"]
    for m in MODELS:
        mt = metrics(g[ACT], g[m])
        rq2_period.append({"period": plabel.replace("\n", " "), "model": m, **mt,
                           "skill": 1 - mt["RMSE"] / pr_rmse})
rq2_period = pd.DataFrame(rq2_period)
savetab(rq2_period, "T6b_rq2_heldout_by_period")

# 5b. Matched comparison: same 3 districts, same dates (>= 2024-10-05);
#     'seen' = chronological models (districts in training), 'unseen' = district-holdout models.
seen = chron_full[chron_full[DIST].isin(HELDOUT)].sort_values([DIST, DATE]).reset_index(drop=True)
unseen = dh_full[dh_full[DATE] >= CS].sort_values([DIST, DATE]).reset_index(drop=True)
if not (len(seen) == len(unseen) and seen[[DIST, DATE]].equals(unseen[[DIST, DATE]])
        and np.allclose(seen[ACT], unseen[ACT], rtol=0, atol=1e-9)):
    raise RuntimeError("Matched seen/unseen comparison rows are not identical.")
checks["matched_comparison_rows"] = int(len(seen))

y_m = seen[ACT].to_numpy(float)
bid_m = block_ids(seen[DATE])
B_m = int(bid_m.max()) + 1
W_m = make_W(B_m, SEED + 100)
matched = []
for m in MODELS:
    s_seen = stats_from_W(block_sums(y_m, seen[m].to_numpy(float), bid_m, B_m), W_m)
    s_un = stats_from_W(block_sums(y_m, unseen[m].to_numpy(float), bid_m, B_m), W_m)
    ms, mu = metrics(y_m, seen[m]), metrics(y_m, unseen[m])
    d_rmse, d_mae = s_un["RMSE"] - s_seen["RMSE"], s_un["MAE"] - s_seen["MAE"]
    matched.append({
        "model": m, "n_rows": len(seen),
        "RMSE_seen": ms["RMSE"], "RMSE_seen_lo": get_ci(s_seen["RMSE"])[0], "RMSE_seen_hi": get_ci(s_seen["RMSE"])[1],
        "RMSE_unseen": mu["RMSE"], "RMSE_unseen_lo": get_ci(s_un["RMSE"])[0], "RMSE_unseen_hi": get_ci(s_un["RMSE"])[1],
        "dRMSE_unseen_minus_seen": mu["RMSE"] - ms["RMSE"],
        "dRMSE_ci_lo": get_ci(d_rmse)[0], "dRMSE_ci_hi": get_ci(d_rmse)[1],
        "rel_change_pct": 100 * (mu["RMSE"] / ms["RMSE"] - 1),
        "MAE_seen": ms["MAE"], "MAE_unseen": mu["MAE"],
        "dMAE_unseen_minus_seen": mu["MAE"] - ms["MAE"],
        "dMAE_ci_lo": get_ci(d_mae)[0], "dMAE_ci_hi": get_ci(d_mae)[1],
        "R2_seen": ms["R2"], "R2_unseen": mu["R2"]})
matched = pd.DataFrame(matched)
savetab(matched, "T6c_rq2_matched_seen_vs_unseen")
assert abs(matched.loc[matched.model == "persistence", "dRMSE_unseen_minus_seen"].iloc[0]) < 1e-9

# 5c. Per-district performance on the chronological test (all 13 districts)
t6d = pd.DataFrame([{"district": dname, "model": m, **metrics(g[ACT], g[m])}
                    for dname, g in chron_full.groupby(DIST) for m in MODELS])
savetab(t6d, "T6d_chronological_by_district")

# 5d. Seasonal error and error by PM2.5 level (chronological test)
chron_full["month"] = chron_full[DATE].dt.month
seasonal = pd.DataFrame([{"month": mo, "model": m, **metrics(g[ACT], g[m])}
                         for mo, g in chron_full.groupby("month") for m in MODELS])
savetab(seasonal, "T7_chronological_error_by_month")

chron_full["level_bin"] = pd.qcut(chron_full[ACT], 10, duplicates="drop")
level = pd.DataFrame([{"bin": str(b), "mean_actual": g[ACT].mean(), "model": m,
                       "RMSE": metrics(g[ACT], g[m])["RMSE"],
                       "mean_residual": metrics(g[ACT], g[m])["mean_residual"]}
                      for b, g in chron_full.groupby("level_bin", observed=True) for m in MODELS])
savetab(level, "T8_chronological_error_by_pm25_level")

# -------------------------------------------------------------------------------------
# 6. RQ3: PERMUTATION IMPORTANCE AND RANKING STABILITY
# -------------------------------------------------------------------------------------
FEATURE_GROUPS = {
    "PM2.5 lags": [f for f in FEATURES if f.startswith("pm25_lag_")],
    "PM2.5 rolling mean/std": [f for f in FEATURES if f.startswith("pm25_roll_")],
    "Wind speed (lag/rolling)": [f for f in FEATURES if f.startswith("weather_")],
    "Calendar (sin/cos)": [f for f in FEATURES if f.startswith(("day_of_", "month_"))],
}
assert sorted(sum(FEATURE_GROUPS.values(), [])) == sorted(FEATURES)

modeling = pd.read_csv(PREP_DIR / "modeling_dataset_nextday_pm25.csv")
modeling[DATE] = parse_dates(modeling[DATE])
REGIME_TEST = {}
for seed in (42, 123, 2025):
    man = pd.read_csv(TRAIN_DIR / "split_manifests" / f"random_seed_{seed}_split.csv")
    pos = man.loc[man.partition == "test", "modeling_row_position"].to_numpy()
    REGIME_TEST[f"random_seed_{seed}"] = modeling.iloc[pos].reset_index(drop=True)
for exp, fn in (("chronological", "chronological_test.csv"),
                ("district_holdout", "district_holdout_test.csv")):
    d = pd.read_csv(PREP_DIR / fn)
    d[DATE] = parse_dates(d[DATE])
    REGIME_TEST[exp] = d


def get_X(df):
    return df[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64").replace(
        [np.inf, -np.inf], np.nan)


def rmse_fn(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def perm_importance(model, X, y, groups, n_repeats, seed):
    rng = np.random.default_rng(seed)
    base = rmse_fn(y, model.predict(X))
    out = []
    for name, cols in groups.items():
        orig = X[cols].to_numpy().copy()
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[cols] = orig[perm]       # rows permuted jointly within a group
            deltas.append(rmse_fn(y, model.predict(Xp)) - base)
        out.append((name, float(np.mean(deltas)),
                    float(np.std(deltas, ddof=1)) if n_repeats > 1 else 0.0))
    return base, out


imp_path = TAB_DIR / "permutation_importance_long.csv"
if REUSE_IMPORTANCE and imp_path.exists():
    imp_long = pd.read_csv(imp_path)
    print("Reusing saved permutation importance table.")
else:
    recs = []
    for ri, regime in enumerate(EXPERIMENTS):
        df = REGIME_TEST[regime]
        X, y = get_X(df), df[TARGET].to_numpy(float)
        for mi, m in enumerate(LEARNED):
            t0 = time.time()
            model = joblib.load(TRAIN_DIR / "models" / f"{regime}__{m}.joblib")
            base, ind = perm_importance(model, X, y, {f: [f] for f in FEATURES},
                                        N_PERM_REPEATS, SEED + 100 * ri + mi)
            _, grp = perm_importance(model, X, y, FEATURE_GROUPS,
                                     N_PERM_REPEATS, SEED + 1000 + 100 * ri + mi)
            ref_rmse = TESTROW[(regime, m)].RMSE
            if abs(base - ref_rmse) > 1e-6:
                print(f"  WARNING: baseline RMSE {base:.6f} != saved {ref_rmse:.6f} ({regime}/{m})")
            for level_name, items in (("feature", ind), ("group", grp)):
                for name, mu, sd in items:
                    recs.append({"regime": regime, "model": m, "level": level_name, "name": name,
                                 "delta_rmse_mean": mu, "delta_rmse_sd": sd,
                                 "baseline_rmse": base, "n_repeats": N_PERM_REPEATS})
            print(f"  permutation importance done: {regime:17s} {m:14s} ({time.time() - t0:5.1f}s)")
    imp_long = pd.DataFrame(recs)
    savetab(imp_long, "permutation_importance_long")

imp_feat = {m: imp_long[(imp_long.model == m) & (imp_long.level == "feature")]
            .pivot(index="name", columns="regime", values="delta_rmse_mean")
            .loc[FEATURES, EXPERIMENTS] for m in LEARNED}
imp_grp = {m: imp_long[(imp_long.model == m) & (imp_long.level == "group")]
           .pivot(index="name", columns="regime", values="delta_rmse_mean")
           .loc[list(FEATURE_GROUPS), EXPERIMENTS] for m in LEARNED}

spearman = {m: imp_feat[m].corr(method="spearman") for m in LEARNED}


def jaccard_top(mat, a, b, k=5):
    sa, sb = set(mat[a].nlargest(k).index), set(mat[b].nlargest(k).index)
    return len(sa & sb) / len(sa | sb)


PAIR_TYPES = {
    "random vs random (3 seeds)": list(combinations(RANDOM_EXPS, 2)),
    "random vs chronological": [(r, "chronological") for r in RANDOM_EXPS],
    "random vs district hold-out": [(r, "district_holdout") for r in RANDOM_EXPS],
    "chronological vs district hold-out": [("chronological", "district_holdout")],
}
stab_rows = []
for m in LEARNED:
    for ptype, pairs in PAIR_TYPES.items():
        stab_rows.append({
            "model": m, "pair_type": ptype,
            "mean_spearman_feature_ranks": float(np.mean([spearman[m].loc[a, b] for a, b in pairs])),
            "mean_top5_jaccard": float(np.mean([jaccard_top(imp_feat[m], a, b) for a, b in pairs]))})
stability = pd.DataFrame(stab_rows)
savetab(stability, "T9_importance_ranking_stability")

cross_model = pd.DataFrame([
    {"regime": r, "pair": f"{a} vs {b}",
     "spearman": float(imp_feat[a][r].corr(imp_feat[b][r], method="spearman"))}
    for r in EXPERIMENTS for a, b in combinations(LEARNED, 2)])
savetab(cross_model, "T9b_importance_rank_agreement_across_models")

grp_tab = imp_long[imp_long.level == "group"].copy()
grp_tab["rank_within_regime_model"] = grp_tab.groupby(["regime", "model"])["delta_rmse_mean"].rank(
    ascending=False)
savetab(grp_tab, "T9c_group_importance_with_ranks")

# -------------------------------------------------------------------------------------
# 7. MAIN PUBLICATION TABLE
# -------------------------------------------------------------------------------------
def fmt_ms(mu, sd, d=2):
    return f"{mu:.{d}f} ± {sd:.{d}f}"


def fmt_ci(p, lo, hi, d=2):
    return f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


t1rows = []
for m in MODELS:
    rec = {"Model": MODEL_LABEL[m]}
    for key, lab, d in (("MAE", "MAE", 2), ("RMSE", "RMSE", 2), ("R2", "R²", 3)):
        vals = [getattr(TESTROW[(e, m)], key) for e in RANDOM_EXPS]
        rec[f"Random (3 seeds) | {lab}"] = fmt_ms(np.mean(vals), np.std(vals, ddof=1), d)
    for exp, tag in (("chronological", "Chronological"), ("district_holdout", "District hold-out")):
        for key, lab, d in (("MAE", "MAE", 2), ("RMSE", "RMSE", 2), ("R2", "R²", 3)):
            rec[f"{tag} | {lab}"] = fmt_ci(*point_ci(exp, m, key), d=d)
    t1rows.append(rec)
table1 = pd.DataFrame(t1rows)
savetab(table1, "TABLE1_test_performance_formatted")
try:
    (TAB_DIR / "TABLE1_test_performance_formatted.tex").write_text(table1.to_latex(index=False, escape=True))
except Exception as exc:
    print("LaTeX export skipped:", exc)

# -------------------------------------------------------------------------------------
# 8. FIGURES
# -------------------------------------------------------------------------------------
def heatmap(ax, mat, xlabels, ylabels, cmap, fmt=".2f", vmin=None, vmax=None):
    mat = np.asarray(mat, float)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=40, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    lo = np.nanmin(mat) if vmin is None else vmin
    hi = np.nanmax(mat) if vmax is None else vmax
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=6.5,
                    color="white" if mat[i, j] > (lo + hi) / 2 else "black")
    ax.grid(False)
    return im


# Fig 1: main test performance
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
panels = [("RMSE", "RMSE (µg/m³)", MODELS), ("MAE", "MAE (µg/m³)", MODELS),
          ("skill", "RMSE skill vs persistence", LEARNED)]
cond_labels = ["Random split\n(mean of 3 seeds)", "Chronological", "District\nhold-out"]
for ax, (key, ylab, mods) in zip(axes, panels):
    w = 0.8 / len(mods)
    for j, m in enumerate(mods):
        off = (j - (len(mods) - 1) / 2) * w
        vals = [point_ci(e, m, key)[0] for e in RANDOM_EXPS]
        mu, sd = np.mean(vals), np.std(vals, ddof=1)
        ax.bar(off, mu, w * 0.95, color=COL[m], label=MODEL_LABEL[m])
        ax.errorbar(off, mu, yerr=sd, color="k", capsize=2, lw=0.8)
        ax.scatter(np.full(3, off), vals, s=6, color="k", zorder=3)
        for i, e in ((1, "chronological"), (2, "district_holdout")):
            pt, lo, hi = point_ci(e, m, key)
            ax.bar(i + off, pt, w * 0.95, color=COL[m])
            ax.errorbar(i + off, pt, yerr=[[max(pt - lo, 0)], [max(hi - pt, 0)]],
                        color="k", capsize=2, lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel(ylab)
    ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False)
fig.suptitle("Test-set performance by evaluation regime. Error bars: SD across seeds (random); "
             "95% block-bootstrap CI (others). Dots: individual seeds.", y=-0.02, fontsize=8)
savefig(fig, "fig1_test_performance_by_regime")

# Fig 2: RQ1 decomposition
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
conds = [C_ALL, C_WIN, C_CHR]
cond_cols = ["#56B4E9", "#CC79A7", "#D55E00"]
for ax, (key, ylab, mods) in zip(axes, (("RMSE", "RMSE (µg/m³)", MODELS),
                                        ("skill", "RMSE skill vs persistence", LEARNED))):
    w = 0.8 / 3
    for ci_, (cname, ccol) in enumerate(zip(conds, cond_cols)):
        for j, m in enumerate(mods):
            r = rq1_df[(rq1_df.model == m) & (rq1_df.condition == cname)].iloc[0]
            mu = r[f"{key}_mean"]
            if cname == C_CHR:
                lo_, hi_ = (r["RMSE_lo"], r["RMSE_hi"]) if key == "RMSE" else (r["skill_lo"], r["skill_hi"])
                err = [[max(mu - lo_, 0)], [max(hi_ - mu, 0)]]
            else:
                err = r[f"{key}_sd"]
            ax.bar(j + (ci_ - 1) * w, mu, w * 0.95, color=ccol, label=cname if j == 0 else None)
            ax.errorbar(j + (ci_ - 1) * w, mu, yerr=err, color="k", capsize=2, lw=0.8)
    ax.set_xticks(range(len(mods)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in mods])
    ax.set_ylabel(ylab)
    ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07), frameon=False)
savefig(fig, "fig2_rq1_random_vs_chronological_decomposition")

# Fig 3: RQ2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
ax = axes[0]
w = 0.38
for j, m in enumerate(MODELS):
    r = matched[matched.model == m].iloc[0]
    for k, (tag, hatch) in enumerate((("seen", None), ("unseen", "//"))):
        pt, lo, hi = r[f"RMSE_{tag}"], r[f"RMSE_{tag}_lo"], r[f"RMSE_{tag}_hi"]
        ax.bar(j + (k - 0.5) * w, pt, w * 0.95, color=COL[m], hatch=hatch, edgecolor="white")
        ax.errorbar(j + (k - 0.5) * w, pt, yerr=[[max(pt - lo, 0)], [max(hi - pt, 0)]],
                    color="k", capsize=2, lw=0.8)
ax.set_xticks(range(4))
ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
ax.set_ylabel("RMSE (µg/m³)")
ax.set_title("(a) Same districts, same dates (≥ 2024-10-05):\nsolid = districts seen in training, hatched = unseen")
ax.grid(axis="x", visible=False)
ax = axes[1]
plabels = list(PERIODS)
for pi, plabel in enumerate(plabels):
    for j, m in enumerate(MODELS):
        v = rq2_period[(rq2_period.period == plabel.replace("\n", " ")) & (rq2_period.model == m)]["RMSE"].iloc[0]
        ax.bar(pi + (j - 1.5) * 0.2, v, 0.19, color=COL[m], label=MODEL_LABEL[m] if pi == 0 else None)
ax.set_xticks(range(3))
ax.set_xticklabels(plabels, fontsize=7)
ax.set_ylabel("RMSE on unseen districts (µg/m³)")
ax.set_title("(b) Unseen districts (Dhaka, Narayanganj, Narsingdi) by period")
ax.grid(axis="x", visible=False)
ax.legend(frameon=False)
savefig(fig, "fig3_rq2_unseen_districts")

# Fig 4: per-district RMSE heatmap (chronological test)
piv = t6d.pivot(index="district", columns="model", values="RMSE")[MODELS]
piv = piv.loc[piv["hist_gbm"].sort_values().index]
fig, ax = plt.subplots(figsize=(5.8, 4.6))
im = heatmap(ax, piv.values, [MODEL_LABEL[m] for m in MODELS],
             [d + (" *" if d in HELDOUT else "") for d in piv.index], "YlOrRd")
ax.set_title("Chronological-test RMSE (µg/m³) by district\n(* = districts held out in the spatial experiment)")
fig.colorbar(im, ax=ax, shrink=0.8)
savefig(fig, "fig4_district_rmse_heatmap_chronological")

# Fig 5: observed vs predicted (selected model, validation-chosen)
sel_regimes = ["random_seed_42", "chronological", "district_holdout"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, exp in zip(axes, sel_regimes):
    ref, pr = ALIGN[(exp, "test")]
    sm = SELECTED[exp]
    y, p = ref[ACT].to_numpy(float), pr[sm]
    lim = float(np.percentile(np.r_[y, p], 99.9))
    hb = ax.hexbin(y, p, gridsize=45, bins="log", mincnt=1, cmap="viridis", extent=(0, lim, 0, lim))
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    mt = metrics(y, p)
    ax.set_title(f"{EXP_LABEL[exp]} – {MODEL_LABEL[sm]}\nRMSE {mt['RMSE']:.2f}, MAE {mt['MAE']:.2f}, R² {mt['R2']:.3f}")
    ax.set_xlabel("Actual estimated PM2.5 (µg/m³)")
    ax.set_ylabel("Predicted (µg/m³)")
    ax.grid(False)
fig.colorbar(hb, ax=axes, shrink=0.8, label="log10(count)")
savefig(fig, "fig5_observed_vs_predicted")

# Fig 6: residual diagnostics
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for k, exp in enumerate(sel_regimes):
    ref, pr = ALIGN[(exp, "test")]
    sm = SELECTED[exp]
    y = ref[ACT].to_numpy(float)
    r_m, r_p = y - pr[sm], y - pr["persistence"]
    ax = axes[0, k]
    bins = np.linspace(-80, 80, 81)
    ax.hist(r_m, bins=bins, color=COL[sm], alpha=0.75, label=MODEL_LABEL[sm])
    ax.hist(r_p, bins=bins, histtype="step", color=COL["persistence"], lw=1.3, label="Persistence")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{EXP_LABEL[exp]}: residual distribution")
    ax.set_xlabel("Residual = actual − predicted (µg/m³)")
    ax.legend(frameon=False)
    ax = axes[1, k]
    p = pr[sm]
    hb = ax.hexbin(p, r_m, gridsize=40, bins="log", mincnt=1, cmap="Greys")
    q = pd.qcut(p, 15, duplicates="drop")
    gb = pd.DataFrame({"p": p, "r": r_m}).groupby(q, observed=True).mean()
    ax.plot(gb["p"], gb["r"], color="red", lw=1.6, label="Binned mean residual")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Predicted (µg/m³)")
    ax.set_ylabel("Residual (µg/m³)")
    ax.set_title(f"{EXP_LABEL[exp]}: residual vs predicted")
    ax.legend(frameon=False)
    ax.grid(False)
fig.tight_layout()
savefig(fig, "fig6_residual_diagnostics")

# Fig 7: chronological test time series and seasonal error
ref, pr = ALIGN[("chronological", "test")]
sm = SELECTED["chronological"]
ts = ref.copy()
ts["persistence"], ts["selected"] = pr["persistence"], pr[sm]
daily = ts.groupby(DATE)[[ACT, "persistence", "selected"]].mean()
fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1.2, 1]})
ax = axes[0]
ax.plot(daily.index, daily[ACT], color="k", lw=1.2, label="Actual (mean of 13 districts)")
ax.plot(daily.index, daily["persistence"], color=COL["persistence"], lw=0.9, label="Persistence")
ax.plot(daily.index, daily["selected"], color=COL[sm], lw=1.1, label=MODEL_LABEL[sm])
ax.set_ylabel("Estimated PM2.5 (µg/m³)")
ax.set_title("Chronological test period (2024-10-05 to 2025-12-31): division-mean daily series")
ax.legend(frameon=False, ncol=3)
ax = axes[1]
months = sorted(seasonal["month"].unique())
for j, m in enumerate(MODELS):
    v = [seasonal[(seasonal.month == mo) & (seasonal.model == m)]["RMSE"].iloc[0] for mo in months]
    ax.bar(np.arange(len(months)) + (j - 1.5) * 0.2, v, 0.19, color=COL[m], label=MODEL_LABEL[m])
ax.set_xticks(range(len(months)))
ax.set_xticklabels([pd.Timestamp(2000, mo, 1).strftime("%b") for mo in months])
ax.set_ylabel("RMSE (µg/m³)")
ax.set_title("Chronological-test RMSE by calendar month")
ax.legend(frameon=False, ncol=4)
ax.grid(axis="x", visible=False)
fig.tight_layout()
savefig(fig, "fig7_chronological_time_series_and_seasonality")

# Fig 8: error by PM2.5 level
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for m in MODELS:
    g = level[level.model == m]
    axes[0].plot(g["mean_actual"], g["RMSE"], marker="o", ms=3, color=COL[m], label=MODEL_LABEL[m])
    axes[1].plot(g["mean_actual"], g["mean_residual"], marker="o", ms=3, color=COL[m], label=MODEL_LABEL[m])
axes[0].set_ylabel("RMSE (µg/m³)")
axes[1].set_ylabel("Mean residual (actual − predicted)")
axes[1].axhline(0, color="k", lw=0.8)
for ax in axes:
    ax.set_xlabel("Mean actual PM2.5 within decile bin (µg/m³)")
axes[0].legend(frameon=False)
fig.suptitle("Chronological test: error by actual PM2.5 level (deciles)", y=1.02)
savefig(fig, "fig8_error_by_pm25_level")

# Fig 9: permutation importance heatmaps (features x regimes)
reg_lab = ["Random\ns42", "Random\ns123", "Random\ns2025", "Chrono-\nlogical", "District\nhold-out"]
fig, axes = plt.subplots(1, 3, figsize=(17, 6.3))
for ax, m in zip(axes, LEARNED):
    mat = imp_feat[m].copy()
    mat = mat.loc[mat.mean(axis=1).sort_values(ascending=False).index]
    im = heatmap(ax, mat.values, reg_lab, [short_name(f) for f in mat.index], "YlGnBu")
    ax.set_title(f"{MODEL_LABEL[m]}")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Increase in RMSE (µg/m³)")
fig.suptitle("Permutation importance of individual features on each regime's test set", y=1.02)
fig.tight_layout()
savefig(fig, "fig9_permutation_importance_features")

# Fig 10: group importance
gcols = ["#0072B2", "#009E73", "#E69F00", "#CC79A7"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), sharey=False)
for ax, m in zip(axes, LEARNED):
    for gi, gname in enumerate(FEATURE_GROUPS):
        sub = imp_long[(imp_long.model == m) & (imp_long.level == "group") & (imp_long.name == gname)] \
            .set_index("regime").loc[EXPERIMENTS]
        ax.bar(np.arange(5) + (gi - 1.5) * 0.2, sub["delta_rmse_mean"], 0.19, color=gcols[gi],
               yerr=sub["delta_rmse_sd"], capsize=1.5, label=gname)
    ax.set_xticks(range(5))
    ax.set_xticklabels(reg_lab)
    ax.set_title(MODEL_LABEL[m])
    ax.set_ylabel("Increase in RMSE (µg/m³)")
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False, fontsize=7)
fig.suptitle("Grouped permutation importance (features permuted jointly within group)", y=1.02)
fig.tight_layout()
savefig(fig, "fig10_permutation_importance_groups")

# Fig 11: ranking stability heatmaps
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, m in zip(axes, LEARNED):
    im = heatmap(ax, spearman[m].values, reg_lab, [s.replace("\n", " ") for s in reg_lab],
                 "RdYlBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{MODEL_LABEL[m]}: Spearman correlation of\nfeature-importance ranks")
fig.colorbar(im, ax=axes, shrink=0.8)
savefig(fig, "fig11_importance_rank_stability")

# -------------------------------------------------------------------------------------
# 9. SUMMARY JSON, CONSOLE REPORT, ARCHIVE
# -------------------------------------------------------------------------------------
top_feats = {f"{r}|{m}": [short_name(x) for x in imp_feat[m][r].nlargest(3).index]
             for r in EXPERIMENTS for m in LEARNED}
summary = {
    "integrity_checks": checks,
    "selected_model_by_validation_rmse": SELECTED,
    "settings": {"n_boot": N_BOOT, "block_days": BLOCK_DAYS, "perm_repeats": N_PERM_REPEATS,
                 "seed": SEED, "target": "estimated PM2.5 (CAMS/ERA5-derived), ug/m3"},
    "rq1_gap_summary": rq1_gap.to_dict("records"),
    "rq2_matched_seen_vs_unseen": matched.to_dict("records"),
    "rq3_ranking_stability": stability.to_dict("records"),
    "rq3_top3_features_by_regime_model": top_feats,
}
with open(EVAL_DIR / "evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)

print("\n" + "=" * 110 + "\nTABLE 1. Test performance (µg/m³)\n" + "=" * 110)
display(table1)
print("\nT3. Test metrics with 95% block-bootstrap CIs")
display(t3.round(3))
print("\nT4. Paired comparisons (dRMSE < 0 means first model is better)")
display(t4.round(3))
print("\nRQ1 gap summary")
display(rq1_gap.round(3))
print("\nRQ2 matched seen-vs-unseen comparison (same 3 districts, same dates)")
display(matched.round(3))
print("\nRQ2 held-out districts by period")
display(rq2_period[["period", "model", "n", "MAE", "RMSE", "R2", "skill"]].round(3))
print("\nRQ3 ranking stability")
display(stability.round(3))
print("\nRQ3 grouped permutation importance (mean increase in RMSE, µg/m³)")
for m in LEARNED:
    print(f"\n{MODEL_LABEL[m]}")
    display(imp_grp[m].round(3))
print("\nRQ3 top-3 individual features per regime/model")
for k, v in top_feats.items():
    print(f"  {k:32s} {v}")

zip_path = shutil.make_archive("/kaggle/working/evaluation_outputs", "zip", root_dir=EVAL_DIR)
print("\nSaved:", EVAL_DIR)
print("ZIP archive:", zip_path)
print(f"Evaluation finished in {(time.time() - T_START) / 60:.1f} minutes.")