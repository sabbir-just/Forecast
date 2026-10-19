Cell 1:


%%writefile /kaggle/working/priority_evaluations.py
# =====================================================================================
# PRIORITY EVALUATIONS 1-8 (reviewer checklist)
# Next-day ESTIMATED PM2.5 (CAMS/ERA5-derived), Dhaka Division.
#
# Reuses: revised_experiment_outputs/all_revised_predictions.csv, pretraining_outputs/*,
#         saved .joblib models. Re-trains only where the item requires it (2, 6, 7).
#
# 1 Paired same-window random vs chronological (14-day moving-block bootstrap)
# 2 Multiple expanding-window chronological tests
# 3 Seasonal-naive (+ training-only climatology) baselines
# 4 Block-preserving permutation importance vs row-wise
# 5 Humidity uncertainty (paired RMSE differences, CIs, win counts)
# 6 Hyperparameter sensitivity (selection on validation only)
# 7 Spatial-independence diagnostics + buffered / grouped hold-outs
# 8 High-pollution event analysis
#
# Run a subset:  PRIORITY_RUN="1,3,5" python priority_evaluations.py
# All metrics evaluate agreement with the model-derived target, NOT ground monitors.
# =====================================================================================

import os, re, json, time, zipfile, warnings, traceback, sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

warnings.filterwarnings("ignore")
T0 = time.time()

# ----------------------------------- PATHS -------------------------------------------
DATASET_ROOT = Path(
    "/kaggle/input/datasets/syedazmulhasansabbir/data-science-project/"
    "A District-Level CAMS Forecast–ERA5 Reanalysis PM-/"
    "dhaka_division_pm_aqi_proxy_dataset/v1.0.0")
PREP_DIR = Path("/kaggle/working/pretraining_outputs")
TRAIN_DIR = Path("/kaggle/working/training_outputs")
REV_DIR = Path("/kaggle/working/revised_experiment_outputs")
OLD_MODEL_ROOT = Path("/kaggle/working/training_outputs/models")
OUT_DIR = Path("/kaggle/working/priority_evaluation_outputs")
TAB, FIG = OUT_DIR / "tables", OUT_DIR / "figures"
for _d in (OUT_DIR, TAB, FIG):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------- SETTINGS ----------------------------------------
RUN = [int(x) for x in os.environ.get("PRIORITY_RUN", "1,2,3,4,5,6,7,8").split(",") if x.strip()]
SEED = 2048
N_BOOT = 2000
BLOCK_DAYS = 14
RANDOM_SEEDS = [42, 123, 2025]

DATE, DIST = "target_date", "district"
TARGET = "target_pm25_ug_m3"
ACT, PRED = "actual_pm25_ug_m3", "predicted_pm25_ug_m3"

TRAIN_END, VAL_START, VAL_END = pd.Timestamp("2023-07-09"), pd.Timestamp("2023-07-10"), pd.Timestamp("2024-10-04")
TEST_START, TEST_END = pd.Timestamp("2024-10-05"), pd.Timestamp("2025-12-31")

BASE_FEATURES = [
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d", "pm25_lag_14d",
    "pm25_roll_mean_prev_3d", "pm25_roll_mean_prev_7d", "pm25_roll_mean_prev_14d",
    "pm25_roll_std_prev_7d",
    "weather_0_wind_speed_mean_ms_lag_1d",
    "weather_0_wind_speed_mean_ms_roll_mean_prev_3d",
    "weather_0_wind_speed_mean_ms_roll_mean_prev_7d",
    "day_of_year_sin", "day_of_year_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos"]
HUMIDITY_FEATURES = ["relative_humidity_lag_1d", "relative_humidity_roll_mean_prev_3d",
                     "relative_humidity_roll_mean_prev_7d"]
FEATURE_SETS = {"original_18": BASE_FEATURES, "corrected_plus_humidity": BASE_FEATURES + HUMIDITY_FEATURES}
MODELS = ["persistence", "ridge", "random_forest", "hist_gbm"]
LEARNED = MODELS[1:]
LABEL = {"persistence": "Persistence", "ridge": "Ridge", "random_forest": "Random Forest",
         "hist_gbm": "HistGBM", "seasonal_naive_7d": "Seasonal naive (t-7)", "climatology": "Climatology"}
COL = {"persistence": "#777777", "ridge": "#E69F00", "random_forest": "#009E73", "hist_gbm": "#0072B2",
       "seasonal_naive_7d": "#CC79A7", "climatology": "#D55E00"}

# Item 2
WINDOW_FIRST_START = "2022-07-01"
WINDOW_MONTHS = 6
WINDOW_FEATURE_SETS = ["original_18"]          # add "corrected_plus_humidity" to also run it
# Item 4
N_PERM_REPEATS = 10
PERM_BLOCK_DAYS = [14]
# Item 7
GROUP_K = 4
BUFFER_KM = 40.0
# Item 8
ABS_THRESHOLDS = {"35.5 (US AQI USG lower bound)": 35.5, "65 (Bangladesh 24-h standard)": 65.0}
TRAIN_QUANTILES = [0.90, 0.95]
MIN_EVENTS = 30

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.22})


# ----------------------------------- HELPERS -----------------------------------------
def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def savecsv(df, name):
    df.to_csv(TAB / f"{name}.csv", index=False)


def savefig(fig, stem):
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def norm_dates(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def read_split(name, folder=None):
    d = pd.read_csv((folder or PREP_DIR) / name, low_memory=False)
    d[DATE] = norm_dates(d[DATE])
    d[DIST] = d[DIST].astype(str).str.strip()
    return d


def get_X(df, feats):
    return df[feats].apply(pd.to_numeric, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    sst = np.sum((y - y.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / sst) if sst > 0 else np.nan,
            "bias_actual_minus_pred": float(np.mean(e))}


def ci(a):
    a = np.asarray(a, float)
    return float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))


class DateBoot:
    """Overlapping moving-block bootstrap over consecutive CALENDAR days.
    All rows that share a date (all districts) are resampled together. Days without
    rows (e.g. random-split test sets) simply carry no weight."""

    def __init__(self, dates, block_days=BLOCK_DAYS, reps=N_BOOT, seed=SEED):
        d = pd.to_datetime(pd.Series(np.asarray(dates))).dt.normalize()
        d0 = d.min()
        self.n_days = int((d.max() - d0).days) + 1
        self.day_idx = (d - d0).dt.days.to_numpy()
        L = max(1, min(int(block_days), self.n_days))
        nb = int(np.ceil(self.n_days / L))
        rng = np.random.default_rng(seed)
        starts = rng.integers(0, self.n_days - L + 1, size=(reps, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(reps, -1)[:, :self.n_days]
        self.W = np.zeros((reps, self.n_days))
        for r in range(reps):
            self.W[r] = np.bincount(idx[r], minlength=self.n_days)

    def draw(self, arr):
        s = np.bincount(self.day_idx, weights=np.asarray(arr, float), minlength=self.n_days)
        return self.W @ s


def boot_metrics(bt, y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    n, sae, sse = bt.draw(np.ones(len(y))), bt.draw(np.abs(e)), bt.draw(e * e)
    sy, syy = bt.draw(y), bt.draw(y * y)
    return {"MAE": sae / n, "RMSE": np.sqrt(sse / n), "R2": 1 - sse / (syy - sy ** 2 / n)}


def panel_table(df, pred_cols, baselines, seed, extra=None):
    """Point metrics + block-bootstrap CIs + RMSE skill vs each baseline (paired resamples)."""
    extra = extra or {}
    bt = DateBoot(df[DATE], seed=seed)
    y = df[ACT].to_numpy(float)
    B = {c: boot_metrics(bt, y, df[c].to_numpy(float)) for c in pred_cols}
    P = {c: metrics(y, df[c].to_numpy(float)) for c in pred_cols}
    rows = []
    for c in pred_cols:
        rec = {**extra, "model": c, "n": len(y), "n_dates": int(df[DATE].nunique())}
        for k in ("MAE", "RMSE", "R2"):
            lo, hi = ci(B[c][k])
            rec[k], rec[k + "_lo"], rec[k + "_hi"] = P[c][k], lo, hi
        for b in baselines:
            if c == b:
                continue
            sk = 1 - B[c]["RMSE"] / B[b]["RMSE"]
            lo, hi = ci(sk)
            rec[f"skill_vs_{b}"] = 1 - P[c]["RMSE"] / P[b]["RMSE"]
            rec[f"skill_vs_{b}_lo"], rec[f"skill_vs_{b}_hi"] = lo, hi
        rows.append(rec)
    return pd.DataFrame(rows)


def build_estimator(name, seed=42, **hp):
    imp = SimpleImputer(strategy="median")
    if name == "ridge":
        return Pipeline([("imputer", imp), ("scaler", StandardScaler()),
                         ("model", Ridge(alpha=hp.get("alpha", 1.0)))])
    if name == "random_forest":
        p = dict(n_estimators=200, max_depth=15, min_samples_leaf=5, n_jobs=-1)
        p.update(hp)
        return Pipeline([("imputer", imp), ("model", RandomForestRegressor(random_state=seed, **p))])
    if name == "hist_gbm":
        p = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False)
        p.update(hp)
        return Pipeline([("imputer", imp), ("model", HistGradientBoostingRegressor(random_state=seed, **p))])
    raise ValueError(name)


class Climatology:
    """Training-only day-of-year climatology per district (circular +/-7 day smoothing)."""

    def __init__(self, train, half=7):
        self.g = float(train[TARGET].mean())
        doy = train[DATE].dt.dayofyear.to_numpy() - 1
        y = train[TARGET].to_numpy(float)

        def smooth(d, v):
            s = np.bincount(d, weights=v, minlength=366)
            c = np.bincount(d, minlength=366).astype(float)
            s2 = sum(np.roll(s, k) for k in range(-half, half + 1))
            c2 = sum(np.roll(c, k) for k in range(-half, half + 1))
            return np.where(c2 > 0, s2 / np.maximum(c2, 1), self.g)

        self.pool = smooth(doy, y)
        self.tab = {d: smooth(doy[i], y[i]) for d, i in train.groupby(DIST).indices.items()}

    def predict(self, df):
        doy = df[DATE].dt.dayofyear.to_numpy() - 1
        dist = df[DIST].to_numpy()
        out = np.empty(len(df))
        for d in np.unique(dist):
            m = dist == d
            out[m] = self.tab.get(d, self.pool)[doy[m]]
        return out


def wide_from_preds(preds, exp, fs, models=MODELS, date_min=None, date_max=None):
    sub = preds[(preds.experiment_id == exp) & (preds.partition == "test") & (preds.feature_set == fs)]
    if date_min is not None:
        sub = sub[sub[DATE] >= pd.Timestamp(date_min)]
    if date_max is not None:
        sub = sub[sub[DATE] <= pd.Timestamp(date_max)]
    ref = None
    for m in models:
        one = sub[sub.model_name == m][[DATE, DIST, ACT, PRED]]
        if one.empty or one.duplicated([DATE, DIST]).any():
            raise RuntimeError(f"Missing/duplicate prediction rows for {exp}/{fs}/{m}")
        if ref is None:
            ref = one.rename(columns={PRED: m})
        else:
            one = one.rename(columns={PRED: m, ACT: "_a"})
            ref = ref.merge(one, on=[DATE, DIST], how="inner", validate="one_to_one")
            if not np.allclose(ref[ACT], ref["_a"], rtol=0, atol=1e-8):
                raise RuntimeError(f"Target mismatch in {exp}/{fs}/{m}")
            ref = ref.drop(columns="_a")
    return ref.sort_values([DATE, DIST]).reset_index(drop=True)


def find_model(exp, name):
    for root in (TRAIN_DIR / "models", OLD_MODEL_ROOT):
        p = root / f"{exp}__{name}.joblib"
        if p.exists():
            return p
    if OLD_MODEL_ROOT.exists():
        f = list(OLD_MODEL_ROOT.rglob(f"{exp}__{name}.joblib"))
        if f:
            return f[0]
    raise FileNotFoundError(f"Model {exp}__{name}.joblib not found")


def locate_raw_root():
    if DATASET_ROOT.exists():
        return DATASET_ROOT
    for r in Path("/kaggle/input").rglob("dhaka_division_pm_aqi_proxy_dataset"):
        return r / "v1.0.0" if (r / "v1.0.0").is_dir() else r
    raise FileNotFoundError("Raw dataset folder not found")


def norm_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


# ----------------------------------- LOAD COMMON DATA --------------------------------
log("Loading common tables")
PREDS = pd.read_csv(REV_DIR / "all_revised_predictions.csv", low_memory=False)
PREDS[DATE] = norm_dates(PREDS[DATE])
PREDS[DIST] = PREDS[DIST].astype(str).str.strip()
MODELING = read_split("modeling_dataset_nextday_pm25.csv")
_corr_path = REV_DIR / "modeling_dataset_corrected.csv"
CORRECTED = read_split("modeling_dataset_corrected.csv", REV_DIR) if _corr_path.exists() else None
LAG7 = MODELING[[DIST, DATE, "pm25_lag_7d"]]
log(f"Predictions: {len(PREDS):,} rows | modeling rows: {len(MODELING):,}")


# =====================================================================================
# ITEM 1 - PAIRED SAME-WINDOW RANDOM vs CHRONOLOGICAL
# =====================================================================================
def item1():
    rows = []
    for fs in FEATURE_SETS:
        for m in MODELS:
            c = PREDS[(PREDS.experiment_id == "chronological") & (PREDS.partition == "test") &
                      (PREDS.feature_set == fs) & (PREDS.model_name == m)][[DATE, DIST, ACT, PRED]]
            for seed in RANDOM_SEEDS:
                r = PREDS[(PREDS.experiment_id == f"random_seed_{seed}") & (PREDS.partition == "test") &
                          (PREDS.feature_set == fs) & (PREDS.model_name == m)]
                r = r[(r[DATE] >= TEST_START) & (r[DATE] <= TEST_END)][[DATE, DIST, ACT, PRED]]
                pair = r.merge(c, on=[DATE, DIST], suffixes=("_rand", "_chron"), validate="one_to_one")
                if pair.empty or not np.allclose(pair[ACT + "_rand"], pair[ACT + "_chron"], rtol=0, atol=1e-8):
                    raise RuntimeError(f"Item1 pairing failed {fs}/{m}/{seed}")
                y = pair[ACT + "_rand"].to_numpy(float)
                pr, pc = pair[PRED + "_rand"].to_numpy(float), pair[PRED + "_chron"].to_numpy(float)
                bt = DateBoot(pair[DATE], seed=SEED + seed)
                br, bc = boot_metrics(bt, y, pr), boot_metrics(bt, y, pc)
                d = bc["RMSE"] - br["RMSE"]
                pct = 100 * (bc["RMSE"] / br["RMSE"] - 1)
                mr, mc = metrics(y, pr), metrics(y, pc)
                dl, dh = ci(d)
                pl, ph = ci(pct)
                rows.append({"feature_set": fs, "model": m, "random_seed": seed, "n_paired_rows": len(pair),
                             "n_paired_dates": int(pair[DATE].nunique()),
                             "RMSE_random": mr["RMSE"], "RMSE_chronological": mc["RMSE"],
                             "delta_RMSE_chron_minus_random": mc["RMSE"] - mr["RMSE"],
                             "delta_ci_lo": dl, "delta_ci_hi": dh,
                             "pct_change": 100 * (mc["RMSE"] / mr["RMSE"] - 1), "pct_ci_lo": pl, "pct_ci_hi": ph,
                             "MAE_random": mr["MAE"], "MAE_chronological": mc["MAE"],
                             "R2_random": mr["R2"], "R2_chronological": mc["R2"]})
    t = pd.DataFrame(rows)
    savecsv(t, "item1_paired_same_window_random_vs_chronological")
    agg = t.groupby(["feature_set", "model"]).agg(
        delta_RMSE_mean=("delta_RMSE_chron_minus_random", "mean"), delta_RMSE_sd=("delta_RMSE_chron_minus_random", "std"),
        pct_change_mean=("pct_change", "mean"), n_seeds_ci_excludes_0=("delta_ci_lo", lambda s: 0)).reset_index()
    agg["n_seeds_ci_excludes_0"] = [
        int(((g.delta_ci_lo > 0) | (g.delta_ci_hi < 0)).sum())
        for _, g in t.groupby(["feature_set", "model"])]
    savecsv(agg, "item1_summary_across_seeds")
    pers = t[t.model == "persistence"]["delta_RMSE_chron_minus_random"].abs().max()
    log(f"Item1 sanity: max |delta| for persistence = {pers:.2e} (should be ~0)")

    fig, ax = plt.subplots(figsize=(8, 4))
    o = t[t.feature_set == "original_18"]
    for j, m in enumerate(MODELS):
        g = o[o.model == m]
        x = j + np.linspace(-0.15, 0.15, len(g))
        ax.errorbar(x, g.delta_RMSE_chron_minus_random,
                    yerr=[g.delta_RMSE_chron_minus_random - g.delta_ci_lo, g.delta_ci_hi - g.delta_RMSE_chron_minus_random],
                    fmt="o", color=COL[m], capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[m] for m in MODELS])
    ax.set_ylabel("RMSE(chronological) - RMSE(random), µg/m³")
    ax.set_title("Paired same-window gap (one point per random seed; 95% 14-day block-bootstrap CI)")
    savefig(fig, "fig_item1_paired_gap")


# =====================================================================================
# ITEM 2 - MULTIPLE EXPANDING-WINDOW CHRONOLOGICAL TESTS
# =====================================================================================
def item2():
    data = (CORRECTED if CORRECTED is not None else MODELING).dropna(subset=[TARGET, "pm25_lag_1d"])
    wins, s = [], pd.Timestamp(WINDOW_FIRST_START)
    while s <= TEST_END:
        e = min(s + pd.DateOffset(months=WINDOW_MONTHS) - pd.Timedelta(days=1), TEST_END)
        wins.append((s, e))
        s = e + pd.Timedelta(days=1)
    log(f"Item2: {len(wins)} windows, first {wins[0]}, last {wins[-1]}")
    tabs = []
    for fs in WINDOW_FEATURE_SETS:
        feats = FEATURE_SETS[fs]
        if any(f not in data.columns for f in feats):
            log(f"Item2: skipping {fs} (missing columns)")
            continue
        for wi, (s, e) in enumerate(wins):
            tr = data[data[DATE] < s]
            te = data[(data[DATE] >= s) & (data[DATE] <= e) & data["pm25_lag_7d"].notna()]
            if len(tr) < 1000 or te.empty:
                continue
            df = te[[DATE, DIST, TARGET]].rename(columns={TARGET: ACT}).reset_index(drop=True)
            df["persistence"] = te["pm25_lag_1d"].to_numpy(float)
            df["seasonal_naive_7d"] = te["pm25_lag_7d"].to_numpy(float)
            df["climatology"] = Climatology(tr).predict(te)
            Xtr, Xte = get_X(tr, feats), get_X(te, feats)
            for m in LEARNED:
                est = build_estimator(m, 42)
                est.fit(Xtr, tr[TARGET].to_numpy(float))
                df[m] = est.predict(Xte)
            cols = ["persistence", "seasonal_naive_7d", "climatology"] + LEARNED
            t = panel_table(df, cols, ["persistence", "seasonal_naive_7d"], SEED + wi,
                            {"feature_set": fs, "window_id": wi + 1, "test_start": s.date(), "test_end": e.date(),
                             "n_train": len(tr), "train_end": (s - pd.Timedelta(days=1)).date(),
                             "train_target_mean": float(tr[TARGET].mean()), "test_target_mean": float(df[ACT].mean())})
            tabs.append(t)
            log(f"Item2 window {wi + 1}/{len(wins)} {fs} done")
    res = pd.concat(tabs, ignore_index=True)
    savecsv(res, "item2_expanding_windows_metrics")
    summ = res.groupby(["feature_set", "model"])[
        ["MAE", "RMSE", "R2", "skill_vs_persistence", "skill_vs_seasonal_naive_7d"]].agg(["mean", "std", "min", "max"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    savecsv(summ.reset_index(), "item2_summary_across_windows")

    r = res[res.feature_set == WINDOW_FEATURE_SETS[0]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in ["persistence", "seasonal_naive_7d", "climatology"] + LEARNED:
        g = r[r.model == m].sort_values("window_id")
        axes[0].plot(g.window_id, g.RMSE, marker="o", color=COL[m], label=LABEL[m])
        if m in LEARNED:
            axes[1].plot(g.window_id, g["skill_vs_persistence"], marker="o", color=COL[m], label=LABEL[m])
    axes[0].set_xlabel("Test window (successive, expanding training)"); axes[0].set_ylabel("RMSE (µg/m³)")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].axhline(0, color="k", lw=0.8); axes[1].set_xlabel("Test window"); axes[1].set_ylabel("RMSE skill vs persistence")
    savefig(fig, "fig_item2_expanding_windows")


# =====================================================================================
# ITEM 3 - SEASONAL-NAIVE (+ CLIMATOLOGY) BASELINES
# =====================================================================================
def item3():
    train_files = {"chronological": "chronological_train.csv", "district_holdout": "district_holdout_train.csv"}
    tabs = []
    for i, (exp, trf) in enumerate(train_files.items()):
        clim = Climatology(read_split(trf))
        for fs in FEATURE_SETS:
            w = wide_from_preds(PREDS, exp, fs, date_min=TEST_START, date_max=TEST_END)
            w = w.merge(LAG7, on=[DIST, DATE], how="left", validate="one_to_one")
            n0 = len(w)
            w = w[w["pm25_lag_7d"].notna()].reset_index(drop=True)
            w["seasonal_naive_7d"] = w["pm25_lag_7d"]
            w["climatology"] = clim.predict(w)
            cols = ["persistence", "seasonal_naive_7d", "climatology"] + LEARNED
            t = panel_table(w, cols, ["persistence", "seasonal_naive_7d"], SEED + i,
                            {"experiment": exp, "feature_set": fs, "rows_dropped_lag7_missing": n0 - len(w)})
            tabs.append(t)
    res = pd.concat(tabs, ignore_index=True)
    savecsv(res, "item3_seasonal_naive_climatology_baselines")

    r = res[(res.experiment == "chronological") & (res.feature_set == "original_18")]
    fig, ax = plt.subplots(figsize=(8, 4))
    ms = ["seasonal_naive_7d", "climatology"] + LEARNED
    for j, (b, lab) in enumerate((("persistence", "vs persistence"), ("seasonal_naive_7d", "vs seasonal naive"))):
        for k, m in enumerate(ms):
            g = r[r.model == m]
            if g.empty or f"skill_vs_{b}" not in g or m == b:
                continue
            v = g[f"skill_vs_{b}"].iloc[0]
            lo, hi = g[f"skill_vs_{b}_lo"].iloc[0], g[f"skill_vs_{b}_hi"].iloc[0]
            ax.bar(k + (j - 0.5) * 0.38, v, 0.36, color=COL[m], alpha=1.0 if j == 0 else 0.55,
                   yerr=[[v - lo], [hi - v]], capsize=2, label=lab if k == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([LABEL[m] for m in ms], rotation=20)
    ax.set_ylabel("RMSE skill (solid: vs persistence, light: vs seasonal naive)")
    savefig(fig, "fig_item3_baseline_skill")


# =====================================================================================
# ITEM 4 - BLOCK-PRESERVING PERMUTATION IMPORTANCE
# =====================================================================================
def item4():
    test = read_split("chronological_test.csv")
    feats = BASE_FEATURES
    X, y = get_X(test, feats), test[TARGET].to_numpy(float)
    Xi = X.copy()
    Xi.index = pd.MultiIndex.from_arrays([test[DIST].values, test[DATE].values])
    cal = pd.date_range(test[DATE].min(), test[DATE].max(), freq="D")
    groups = {"PM2.5 lags": [f for f in feats if f.startswith("pm25_lag_")],
              "PM2.5 rolling mean/std": [f for f in feats if f.startswith("pm25_roll_")],
              "Wind speed (lag/rolling)": [f for f in feats if f.startswith("weather_")],
              "Calendar (sin/cos)": [f for f in feats if f.startswith(("day_of_", "month_"))]}
    units = {**{f: [f] for f in feats}, **groups}
    level = {**{f: "feature" for f in feats}, **{g: "group" for g in groups}}

    def row_perm(cols, rng):
        Xp = X.copy()
        Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
        return Xp

    def block_perm(cols, rng, L):
        blocks = [cal[i:i + L].values for i in range(0, len(cal), L)]
        seq = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        src = pd.Series(seq, index=cal).reindex(test[DATE].values).to_numpy()
        keys = pd.MultiIndex.from_arrays([test[DIST].values, src])
        Xp = X.copy()
        Xp[cols] = Xi[cols].reindex(keys).to_numpy()
        return Xp

    modes = [("row", None)] + [(f"block_{L}d", L) for L in PERM_BLOCK_DAYS]
    rec = []
    for mi, m in enumerate(LEARNED):
        model = joblib.load(find_model("chronological", m))
        base = float(np.sqrt(np.mean((y - model.predict(X)) ** 2)))
        for ki, (mode, L) in enumerate(modes):
            rng = np.random.default_rng(SEED + 100 * mi + ki)
            for name, cols in units.items():
                d = []
                for _ in range(N_PERM_REPEATS):
                    Xp = row_perm(cols, rng) if L is None else block_perm(cols, rng, L)
                    d.append(float(np.sqrt(np.mean((y - model.predict(Xp)) ** 2)) - base))
                rec.append({"model": m, "mode": mode, "level": level[name], "name": name, "baseline_rmse": base,
                            "delta_rmse_mean": np.mean(d), "delta_rmse_sd": np.std(d, ddof=1), "n_repeats": N_PERM_REPEATS})
        log(f"Item4 {m} done")
    imp = pd.DataFrame(rec)
    savecsv(imp, "item4_permutation_importance_row_vs_block")

    cmp_rows = []
    for m in LEARNED:
        for lv in ("feature", "group"):
            piv = imp[(imp.model == m) & (imp.level == lv)].pivot(index="name", columns="mode", values="delta_rmse_mean")
            for mode in [x[0] for x in modes[1:]]:
                top = lambda s, k=5: set(s.nlargest(k).index)
                a, b = top(piv["row"]), top(piv[mode])
                cmp_rows.append({"model": m, "level": lv, "block_mode": mode,
                                 "spearman_row_vs_block": float(piv["row"].corr(piv[mode], method="spearman")),
                                 "top5_jaccard": len(a & b) / len(a | b) if lv == "feature" else np.nan,
                                 "top_unit_row": piv["row"].idxmax(), "top_unit_block": piv[mode].idxmax()})
    savecsv(pd.DataFrame(cmp_rows), "item4_row_vs_block_rank_agreement")

    xm = []
    for mode, _ in modes:
        for lv in ("feature", "group"):
            piv = imp[(imp["mode"] == mode) & (imp.level == lv)].pivot(index="name", columns="model", values="delta_rmse_mean")
            for a, b in combinations(LEARNED, 2):
                xm.append({"mode": mode, "level": lv, "pair": f"{a} vs {b}",
                           "spearman": float(piv[a].corr(piv[b], method="spearman"))})
    savecsv(pd.DataFrame(xm), "item4_cross_model_rank_agreement")

    g = imp[imp.level == "group"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, m in zip(axes, LEARNED):
        for k, (mode, _) in enumerate(modes):
            s = g[(g.model == m) & (g["mode"] == mode)].set_index("name").loc[list(groups)]
            ax.bar(np.arange(4) + (k - (len(modes) - 1) / 2) * 0.3, s.delta_rmse_mean, 0.28,
                   yerr=s.delta_rmse_sd, capsize=2, label=mode)
        ax.set_xticks(range(4)); ax.set_xticklabels([x.replace(" (", "\n(") for x in groups], fontsize=7)
        ax.set_title(LABEL[m]); ax.set_ylabel("Increase in RMSE (µg/m³)")
    axes[0].legend(frameon=False)
    savefig(fig, "fig_item4_row_vs_block_importance")


# =====================================================================================
# ITEM 5 - HUMIDITY UNCERTAINTY
# =====================================================================================
def _pair_orig_corr(exp, m):
    s = PREDS[(PREDS.experiment_id == exp) & (PREDS.partition == "test") & (PREDS.model_name == m)]
    o = s[s.feature_set == "original_18"][[DATE, DIST, ACT, PRED]]
    c = s[s.feature_set == "corrected_plus_humidity"][[DATE, DIST, ACT, PRED]]
    p = o.merge(c, on=[DATE, DIST], suffixes=("_o", "_c"), validate="one_to_one")
    if len(p) != len(o) or len(p) != len(c) or not np.allclose(p[ACT + "_o"], p[ACT + "_c"], rtol=0, atol=1e-8):
        raise RuntimeError(f"Humidity pairing failed {exp}/{m}")
    return p


def _hum_row(p, protocol, exp, m, seed):
    y = p[ACT + "_o"].to_numpy(float)
    po, pc = p[PRED + "_o"].to_numpy(float), p[PRED + "_c"].to_numpy(float)
    bt = DateBoot(p[DATE], seed=seed)
    bo, bc = boot_metrics(bt, y, po), boot_metrics(bt, y, pc)
    d, pct = bc["RMSE"] - bo["RMSE"], 100 * (bc["RMSE"] / bo["RMSE"] - 1)
    ro, rc = metrics(y, po)["RMSE"], metrics(y, pc)["RMSE"]
    dl, dh = ci(d)
    pl, ph = ci(pct)
    return {"protocol": protocol, "experiment_id": exp, "model": m, "n": len(p),
            "RMSE_original": ro, "RMSE_humidity": rc, "delta_RMSE": rc - ro, "delta_ci_lo": dl, "delta_ci_hi": dh,
            "pct_change": 100 * (rc / ro - 1), "pct_ci_lo": pl, "pct_ci_hi": ph,
            "humidity_improves": bool(rc < ro), "ci_excludes_0": bool(dl > 0 or dh < 0)}


def item5():
    exps = sorted(PREDS.experiment_id.unique())
    rows = []
    for m in LEARNED:
        for i, exp in enumerate(exps):
            proto = ("random" if exp.startswith("random_seed") else
                     "lodo" if exp.startswith("lodo_") else exp)
            rows.append(_hum_row(_pair_orig_corr(exp, m), proto, exp, m, SEED + i))
        lodo_pairs = [_pair_orig_corr(e, m) for e in exps if e.startswith("lodo_")]
        if lodo_pairs:
            rows.append(_hum_row(pd.concat(lodo_pairs, ignore_index=True), "lodo_pooled", "lodo_all_districts", m, SEED + 999))
        log(f"Item5 {m} done")
    t = pd.DataFrame(rows)
    savecsv(t, "item5_humidity_paired_rmse_differences")
    s = t[t.protocol.isin(["random", "lodo"])].groupby(["protocol", "model"]).agg(
        n_experiments=("delta_RMSE", "size"), n_humidity_improves=("humidity_improves", "sum"),
        mean_delta_RMSE=("delta_RMSE", "mean"), median_delta_RMSE=("delta_RMSE", "median"),
        mean_pct_change=("pct_change", "mean"), n_ci_excludes_0=("ci_excludes_0", "sum")).reset_index()
    savecsv(s, "item5_humidity_win_counts")

    show = t[t.protocol.isin(["random", "chronological", "district_holdout", "lodo_pooled"])]
    labs = list(dict.fromkeys(show.protocol))
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, m in enumerate(LEARNED):
        for j, pr in enumerate(labs):
            g = show[(show.model == m) & (show.protocol == pr)]
            x = j + (k - 1) * 0.25 + (np.linspace(-0.05, 0.05, len(g)) if len(g) > 1 else 0)
            ax.errorbar(x, g.delta_RMSE, yerr=[g.delta_RMSE - g.delta_ci_lo, g.delta_ci_hi - g.delta_RMSE],
                        fmt="o", color=COL[m], capsize=2, label=LABEL[m] if j == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs)
    ax.set_ylabel("RMSE(humidity) - RMSE(original), µg/m³")
    ax.legend(frameon=False)
    savefig(fig, "fig_item5_humidity_effect_ci")


# =====================================================================================
# ITEM 6 - HYPERPARAMETER SENSITIVITY (validation-only selection)
# =====================================================================================
GRIDS = {
    "ridge": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)],
    "random_forest": [{"max_depth": d, "min_samples_leaf": l} for d in (8, 15, None) for l in (5, 20)],
    "hist_gbm": [{"learning_rate": lr, "max_leaf_nodes": nl} for lr in (0.03, 0.05, 0.1) for nl in (15, 31, 63)],
}
DEFAULTS = {"ridge": {"alpha": 1.0}, "random_forest": {"max_depth": 15, "min_samples_leaf": 5},
            "hist_gbm": {"learning_rate": 0.05, "max_leaf_nodes": 31}}
PROTOCOLS = {"random_seed_42": ("random_train.csv", "random_validation.csv", "random_test.csv"),
             "chronological": ("chronological_train.csv", "chronological_validation.csv", "chronological_test.csv"),
             "district_holdout": ("district_holdout_train.csv", "district_holdout_validation.csv", "district_holdout_test.csv")}


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))


def item6():
    rows = []
    for proto, (ftr, fva, fte) in PROTOCOLS.items():
        tr, va, te = read_split(ftr), read_split(fva), read_split(fte)
        win = (te[DATE] >= TEST_START) & (te[DATE] <= TEST_END)
        Xtr, Xva, Xte = (get_X(d, BASE_FEATURES) for d in (tr, va, te))
        ytr, yva, yte = (d[TARGET].to_numpy(float) for d in (tr, va, te))
        rows.append({"protocol": proto, "model": "persistence", "config": "rule: lag_1d", "is_default": True,
                     "val_rmse": _rmse(yva, Xva["pm25_lag_1d"]), "test_rmse": _rmse(yte, Xte["pm25_lag_1d"]),
                     "test_rmse_window": _rmse(yte[win], Xte["pm25_lag_1d"][win])})
        for m, grid in GRIDS.items():
            for hp in grid:
                est = build_estimator(m, 42, **hp)
                est.fit(Xtr, ytr)
                pv, pt = est.predict(Xva), est.predict(Xte)
                rows.append({"protocol": proto, "model": m, "config": json.dumps(hp), "is_default": hp == DEFAULTS[m],
                             "val_rmse": _rmse(yva, pv), "test_rmse": _rmse(yte, pt),
                             "test_rmse_window": _rmse(yte[win], pt[win])})
            log(f"Item6 {proto}/{m} done")
    full = pd.DataFrame(rows)
    full["selected_by_validation"] = False
    for (proto, m), g in full.groupby(["protocol", "model"]):
        full.loc[g["val_rmse"].idxmin(), "selected_by_validation"] = True
    # Full grid with validation scores only (clean), and a separate descriptive file with test scores.
    savecsv(full.drop(columns=["test_rmse", "test_rmse_window"]), "item6_grid_validation_scores")
    savecsv(full, "item6_grid_descriptive_incl_test_DO_NOT_USE_FOR_SELECTION")

    sel = full[full.selected_by_validation].copy()
    dfl = full[full.is_default].copy()
    sel["setting"], dfl["setting"] = "validation_selected", "default"
    final = pd.concat([sel, dfl], ignore_index=True)[
        ["setting", "protocol", "model", "config", "val_rmse", "test_rmse", "test_rmse_window"]]
    savecsv(final, "item6_selected_vs_default_test")

    gap = []
    for setting, g in final.groupby("setting"):
        for m in MODELS:
            q = g[g.model == m].set_index("protocol")
            if len(q) < 3:
                continue
            rnd_all, rnd_win, chron = q.loc["random_seed_42", "test_rmse"], q.loc["random_seed_42", "test_rmse_window"], q.loc["chronological", "test_rmse"]
            gap.append({"setting": setting, "model": m, "RMSE_random_all": rnd_all, "RMSE_random_same_window": rnd_win,
                        "RMSE_chronological": chron, "RMSE_district_holdout": q.loc["district_holdout", "test_rmse"],
                        "chron_vs_random_all_pct": 100 * (chron / rnd_all - 1),
                        "chron_vs_random_window_pct": 100 * (chron / rnd_win - 1)})
    savecsv(pd.DataFrame(gap), "item6_protocol_gap_under_tuning")
    spread = full[full.model != "persistence"].groupby(["protocol", "model"]).agg(
        n_configs=("val_rmse", "size"), val_rmse_min=("val_rmse", "min"), val_rmse_max=("val_rmse", "max"),
        test_rmse_min=("test_rmse", "min"), test_rmse_max=("test_rmse", "max")).reset_index()
    savecsv(spread, "item6_grid_spread_descriptive")


# =====================================================================================
# ITEM 7 - SPATIAL-INDEPENDENCE DIAGNOSTICS + BUFFERED / GROUPED HOLD-OUTS
# =====================================================================================
LAT_RE = re.compile(r"(^|[^a-z])(lat|latitude)([^a-z]|$)")
LON_RE = re.compile(r"(^|[^a-z])(lon|lng|longitude)([^a-z]|$)")


def find_coords():
    root = locate_raw_root()
    supported = (".csv", ".csv.gz", ".parquet", ".xlsx", ".xls")
    skip = ("readme", "dictionary", "manifest", "checksum", "metadata", "license", "description")
    cands = []
    for p in sorted(root.rglob("*")):
        n = p.name.lower()
        if not p.is_file() or not n.endswith(supported) or any(s in n for s in skip):
            continue
        try:
            if n.endswith(".parquet"):
                import pyarrow.parquet as pq
                cols = list(pq.ParquetFile(p).schema.names)
            elif n.endswith((".csv", ".csv.gz")):
                cols = list(pd.read_csv(p, nrows=0).columns)
            else:
                cols = list(pd.read_excel(p, nrows=0).columns)
        except Exception:
            continue
        dc = next((c for c in cols if norm_name(c) in ("district", "districtname", "adm2name", "adm2en", "name2")), None)
        lat = [c for c in cols if LAT_RE.search(str(c).lower())]
        lon = [c for c in cols if LON_RE.search(str(c).lower())]
        if dc and lat and lon:
            cands.append((p.stat().st_size, p, dc, lat, lon))
    if not cands:
        return None
    _, p, dc, lat, lon = sorted(cands, key=lambda z: z[0])[0]
    use = [dc] + lat + lon
    n = p.name.lower()
    raw = (pd.read_parquet(p, columns=use) if n.endswith(".parquet") else
           pd.read_csv(p, usecols=use, low_memory=False) if n.endswith((".csv", ".csv.gz")) else
           pd.read_excel(p, usecols=use))
    raw[dc] = raw[dc].astype(str).str.strip()
    grid_lat = [c for c in lat if any(t in str(c).lower() for t in ("grid", "cams"))]
    grid_lon = [c for c in lon if any(t in str(c).lower() for t in ("grid", "cams"))]
    cen_lat = [c for c in lat if c not in grid_lat] or lat
    cen_lon = [c for c in lon if c not in grid_lon] or lon
    cen = raw.groupby(dc).agg(lat=(cen_lat[0], "median"), lon=(cen_lon[0], "median")).reset_index().rename(columns={dc: DIST})
    gl, go = (grid_lat[0], grid_lon[0]) if grid_lat and grid_lon else (cen_lat[0], cen_lon[0])
    pairs = raw[[dc, gl, go]].dropna().round(4).drop_duplicates()
    pairs["cell"] = pairs[gl].astype(str) + "," + pairs[go].astype(str)
    return {"source": str(p), "centroids": cen, "cells": pairs.rename(columns={dc: DIST})[[DIST, "cell"]],
            "cell_columns": [gl, go]}


def haversine_matrix(lat, lon):
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlat, dlon = la - la.T, lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def item7(preds_all=None):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    daily = pd.read_csv(PREP_DIR / "daily_pm25_dataset.csv", low_memory=False)
    daily["date"] = norm_dates(daily["date"])
    daily["district"] = daily["district"].astype(str).str.strip()
    wide = daily.pivot(index="date", columns="district", values="pm25_ug_m3")
    dists = list(wide.columns)
    c_lvl, c_diff = wide.corr(min_periods=100), wide.diff().corr(min_periods=100)

    info = None
    try:
        info = find_coords()
    except Exception as exc:
        log(f"Item7: coordinate search failed ({exc})")
    D = None
    if info is not None:
        cen = info["centroids"].set_index(DIST).reindex(dists)
        if cen[["lat", "lon"]].notna().all().all():
            D = haversine_matrix(cen["lat"].to_numpy(float), cen["lon"].to_numpy(float))
        savecsv(info["centroids"], "item7_district_coordinates")
        cells = info["cells"].groupby(DIST)["cell"].agg(lambda s: " | ".join(sorted(set(s)))).reset_index()
        cells["n_cells"] = cells["cell"].str.count(r"\|") + 1
        share = info["cells"].groupby("cell")[DIST].agg(lambda s: sorted(set(s)))
        shared = share[share.map(len) > 1]
        cells["shares_cell_with"] = cells[DIST].map(
            lambda d: sorted({x for lst in shared if d in lst for x in lst if x != d}))
        savecsv(cells, "item7_grid_cell_sharing")
        log(f"Item7: coords from {info['source']}; {len(shared)} coordinate cells shared by >1 district")
    else:
        log("Item7: no lat/lon columns found in raw data; using correlation-based proxies only")

    pair_rows = []
    for i, j in combinations(range(len(dists)), 2):
        a, b = wide.iloc[:, i], wide.iloc[:, j]
        ok = a.notna() & b.notna()
        pair_rows.append({"district_a": dists[i], "district_b": dists[j],
                          "corr_level": c_lvl.iloc[i, j], "corr_daily_change": c_diff.iloc[i, j],
                          "frac_days_identical": float((np.abs(a[ok] - b[ok]) < 1e-9).mean()) if ok.any() else np.nan,
                          "distance_km": D[i, j] if D is not None else np.nan})
    pairs = pd.DataFrame(pair_rows)
    savecsv(pairs, "item7_pairwise_similarity")
    if D is not None:
        pairs[["distance_km", "corr_level", "corr_daily_change"]].corr(method="spearman").to_csv(
            TAB / "item7_distance_vs_correlation_spearman.csv")

    # Groupings and neighbour sets
    if D is not None:
        Dm = D.copy()
    else:
        Dm = 1 - c_diff.fillna(0).to_numpy()
        np.fill_diagonal(Dm, 0)
    Dm = (Dm + Dm.T) / 2
    labels = fcluster(linkage(squareform(Dm, checks=False), "average"), GROUP_K, "maxclust")
    grp = pd.DataFrame({DIST: dists, "group": labels})
    savecsv(grp, "item7_geographic_groups")

    def neighbours(d):
        i = dists.index(d)
        if D is not None:
            return {dists[k] for k in range(len(dists)) if k != i and D[i, k] <= BUFFER_KM}
        order = np.argsort(-c_diff.fillna(-1).to_numpy()[i])
        return {dists[k] for k in order[:4] if k != i}

    base = MODELING
    window = (base[DATE] >= TEST_START) & (base[DATE] <= TEST_END)

    def fit_eval(train_d, test_d, extra, seed):
        tr = base[(base[DATE] <= TRAIN_END) & base[DIST].isin(train_d)]
        te = base[window & base[DIST].isin(test_d)]
        df = te[[DATE, DIST, TARGET]].rename(columns={TARGET: ACT}).reset_index(drop=True)
        df["persistence"] = te["pm25_lag_1d"].to_numpy(float)
        Xtr, Xte = get_X(tr, BASE_FEATURES), get_X(te, BASE_FEATURES)
        for m in LEARNED:
            est = build_estimator(m, 42)
            est.fit(Xtr, tr[TARGET].to_numpy(float))
            df[m] = est.predict(Xte)
        return panel_table(df, MODELS, ["persistence"], seed, {**extra, "n_train_rows": len(tr), "n_train_districts": len(set(train_d))})

    tabs = []
    for k, d in enumerate(dists):
        exp = "lodo_" + re.sub(r"[^A-Za-z0-9]+", "_", d).strip("_")
        try:
            w = wide_from_preds(PREDS, exp, "original_18")
            t = panel_table(w, MODELS, ["persistence"], SEED + k,
                            {"scheme": "LODO_standard", "heldout": d, "n_train_districts": len(dists) - 1})
            tabs.append(t)
        except Exception as exc:
            log(f"Item7: standard LODO for {d} unavailable in predictions ({exc}); retraining")
            tabs.append(fit_eval([x for x in dists if x != d], [d], {"scheme": "LODO_standard", "heldout": d}, SEED + k))
        nb = neighbours(d)
        tabs.append(fit_eval([x for x in dists if x != d and x not in nb], [d],
                             {"scheme": "LODO_buffered", "heldout": d, "excluded_neighbours": ";".join(sorted(nb))}, SEED + 50 + k))
        log(f"Item7 LODO {d} done")
    for g in sorted(set(labels)):
        held = [d for d, l in zip(dists, labels) if l == g]
        nb = set().union(*[neighbours(d) for d in held]) - set(held)
        tabs.append(fit_eval([x for x in dists if x not in held], held,
                             {"scheme": "group_holdout", "heldout": f"group{g}:" + ";".join(held)}, SEED + 100 + g))
        tabs.append(fit_eval([x for x in dists if x not in held and x not in nb], held,
                             {"scheme": "group_holdout_buffered", "heldout": f"group{g}:" + ";".join(held),
                              "excluded_neighbours": ";".join(sorted(nb))}, SEED + 200 + g))
        log(f"Item7 group {g} done")
    res = pd.concat(tabs, ignore_index=True)
    savecsv(res, "item7_spatial_holdout_results")
    summ = res.groupby(["scheme", "model"])[["RMSE", "MAE", "R2", "skill_vs_persistence"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    savecsv(summ.reset_index(), "item7_spatial_scheme_summary")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    mat = c_diff.to_numpy()
    im = ax.imshow(mat, cmap="viridis", vmin=np.nanmin(mat), vmax=1)
    ax.set_xticks(range(len(dists))); ax.set_xticklabels(dists, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(dists))); ax.set_yticklabels(dists, fontsize=7)
    ax.grid(False); ax.set_title("Correlation of day-to-day PM2.5 changes between districts")
    fig.colorbar(im, ax=ax, shrink=0.8)
    savefig(fig, "fig_item7_daily_change_correlation")


# =====================================================================================
# ITEM 8 - HIGH-POLLUTION EVENT PERFORMANCE
# =====================================================================================
def item8():
    train = read_split("chronological_train.csv")
    thresholds = dict(ABS_THRESHOLDS)
    for q in TRAIN_QUANTILES:
        thresholds[f"train P{int(q * 100)} ({train[TARGET].quantile(q):.1f})"] = float(train[TARGET].quantile(q))
    rows, decile_rows = [], []
    for fs in FEATURE_SETS:
        w = wide_from_preds(PREDS, "chronological", fs, date_min=TEST_START, date_max=TEST_END)
        w = w.merge(LAG7, on=[DIST, DATE], how="left", validate="one_to_one")
        w = w[w["pm25_lag_7d"].notna()].reset_index(drop=True)
        w["seasonal_naive_7d"] = w["pm25_lag_7d"]
        cols = ["persistence", "seasonal_naive_7d"] + LEARNED
        y = w[ACT].to_numpy(float)
        bt = DateBoot(w[DATE], seed=SEED + 8)
        d_all = bt.draw(np.ones(len(y)))
        for tname, thr in thresholds.items():
            ev = (y >= thr).astype(float)
            n_ev = int(ev.sum())
            for c in cols:
                p = w[c].to_numpy(float)
                pe = (p >= thr).astype(float)
                tp, fp = ev * pe, (1 - ev) * pe
                err = y - p
                ev_n = bt.draw(ev)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rec_b = bt.draw(tp) / ev_n
                    prec_b = np.where(bt.draw(pe) > 0, bt.draw(tp) / bt.draw(pe), np.nan)
                    bias_b = bt.draw(err * ev) / ev_n
                    rmse_b = np.sqrt(bt.draw(err ** 2 * ev) / ev_n)
                    fpr_b = bt.draw(fp) / bt.draw(1 - ev)
                tpn, fpn = float(tp.sum()), float(fp.sum())
                ok = n_ev >= MIN_EVENTS
                rec = tpn / n_ev if n_ev else np.nan
                prec = tpn / (tpn + fpn) if (tpn + fpn) > 0 else np.nan
                r = {"feature_set": fs, "threshold": tname, "threshold_value": thr, "model": c,
                     "n_rows": len(y), "n_events": n_ev, "event_prevalence": n_ev / len(y),
                     "n_predicted_events": int(pe.sum()), "reportable_pr": bool(ok),
                     "recall": rec if ok else np.nan, "miss_rate": 1 - rec if ok and n_ev else np.nan,
                     "precision": prec if ok else np.nan,
                     "F1": 2 * prec * rec / (prec + rec) if ok and prec and rec and (prec + rec) > 0 else np.nan,
                     "false_alarm_rate": fpn / (len(y) - n_ev) if len(y) > n_ev else np.nan,
                     "event_bias_actual_minus_pred": float(np.sum(err * ev) / n_ev) if n_ev else np.nan,
                     "event_RMSE": float(np.sqrt(np.sum(err ** 2 * ev) / n_ev)) if n_ev else np.nan}
                for k, arr in (("recall", rec_b), ("precision", prec_b), ("event_bias", bias_b), ("event_RMSE", rmse_b)):
                    r[k + "_lo"], r[k + "_hi"] = ci(arr) if ok else (np.nan, np.nan)
                rows.append(r)
        top = y >= np.quantile(y, 0.9)
        for c in cols:
            decile_rows.append({"feature_set": fs, "model": c, "top_decile_threshold": float(np.quantile(y, 0.9)),
                                "top_decile_bias_actual_minus_pred": float(np.mean(y[top] - w[c].to_numpy(float)[top])),
                                "top_decile_RMSE": float(np.sqrt(np.mean((y[top] - w[c].to_numpy(float)[top]) ** 2)))})
    res = pd.DataFrame(rows)
    savecsv(res, "item8_event_metrics")
    savecsv(pd.DataFrame(decile_rows), "item8_top_decile_bias")

    r = res[(res.feature_set == "original_18") & res.reportable_pr]
    if not r.empty:
        ths = list(dict.fromkeys(r.threshold))
        fig, ax = plt.subplots(figsize=(10, 4))
        cols = ["persistence", "seasonal_naive_7d"] + LEARNED
        for k, c in enumerate(cols):
            g = r[r.model == c].set_index("threshold").reindex(ths)
            x = np.arange(len(ths)) + (k - (len(cols) - 1) / 2) * 0.14
            ax.errorbar(x, g.recall, yerr=[(g.recall - g.recall_lo).fillna(0), (g.recall_hi - g.recall).fillna(0)],
                        fmt="o", color=COL[c], capsize=2, label=LABEL[c])
        ax.set_xticks(range(len(ths))); ax.set_xticklabels(ths, rotation=15, fontsize=7)
        ax.set_ylabel("Event recall (95% block-bootstrap CI)"); ax.legend(frameon=False, fontsize=7)
        savefig(fig, "fig_item8_event_recall")


# =====================================================================================
# MAIN
# =====================================================================================
ITEMS = {1: item1, 2: item2, 3: item3, 4: item4, 5: item5, 6: item6, 7: item7, 8: item8}
status = {}
for k in sorted(RUN):
    log(f"=== ITEM {k} START ===")
    try:
        ITEMS[k]()
        status[k] = "ok"
    except Exception as exc:
        status[k] = f"FAILED: {exc}"
        log(f"ITEM {k} FAILED: {exc}")
        log(traceback.format_exc())
    log(f"=== ITEM {k} {status[k]} ===")

with open(OUT_DIR / "run_status.json", "w") as f:
    json.dump({"status": status, "n_boot": N_BOOT, "block_days": BLOCK_DAYS,
               "caveat": "Metrics evaluate agreement with CAMS/ERA5-derived estimated PM2.5, not ground monitors."}, f, indent=2)

zip_path = Path("/kaggle/working/priority_evaluation_results.zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for p in OUT_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".csv", ".png", ".pdf", ".json"}:
            zf.write(p, arcname=str(p.relative_to(OUT_DIR)))
print("\nStatus:", status)
print("Archive:", zip_path)
print(f"Finished in {(time.time() - T0) / 60:.1f} min")



Cell 2:
!python /kaggle/working/priority_evaluations.py
# subset, e.g.:  !PRIORITY_RUN="1,3,5" python /kaggle/working/priority_evaluations.py