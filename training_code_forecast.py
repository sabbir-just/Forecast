Cell 1:


%%writefile /kaggle/working/train_models.py
# =====================================================================================
# TRAINING PIPELINE
# Next-day estimated PM2.5 forecasting, Dhaka Division, Bangladesh
# (CAMS forecast + ERA5 reanalysis derived estimates, NOT ground-monitor data)
#
# Models: persistence, ridge, random_forest, hist_gbm  (CPU only, scikit-learn)
# Experiments: random_seed_{42,123,2025}, chronological, district_holdout
# =====================================================================================

import os
import re
import sys
import json
import time
import logging
import platform
import traceback
import zipfile
from pathlib import Path
from importlib import metadata

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------

DATASET_ROOT = (
    "/kaggle/input/datasets/syedazmulhasansabbir/data-science-project/"
    "A District-Level CAMS Forecast–ERA5 Reanalysis PM-/"
    "dhaka_division_pm_aqi_proxy_dataset/v1.0.0"
)

PREP_DIR = Path("/kaggle/working/pretraining_outputs")
OUT_DIR = Path("/kaggle/working/training_outputs")
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"
SPLIT_DIR = OUT_DIR / "split_manifests"

BASE_SEED = 42
RANDOM_SEEDS = [42, 123, 2025]

DATE_COL = "target_date"
DIST_COL = "district"
TARGET_COL = "target_pm25_ug_m3"
METADATA_COLS = [DATE_COL, DIST_COL, TARGET_COL]

EXPECTED_FEATURES = [
    "pm25_lag_1d", "pm25_lag_2d", "pm25_lag_3d", "pm25_lag_7d", "pm25_lag_14d",
    "pm25_roll_mean_prev_3d", "pm25_roll_mean_prev_7d", "pm25_roll_mean_prev_14d",
    "pm25_roll_std_prev_7d",
    "weather_0_wind_speed_mean_ms_lag_1d",
    "weather_0_wind_speed_mean_ms_roll_mean_prev_3d",
    "weather_0_wind_speed_mean_ms_roll_mean_prev_7d",

    "day_of_year_sin", "day_of_year_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
]

CALENDAR_FEATURES = {
    "day_of_year_sin", "day_of_year_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
}

# Strictly historical lag / prior-rolling columns (PM2.5 or meteorological).
HISTORICAL_PATTERN = re.compile(
    r"^.+_(?:lag_\d+d|roll_mean_prev_\d+d|roll_std_prev_\d+d)$"
)

# Verified file inventory: key -> (file name, expected rows)
FILES = {
    "modeling": ("modeling_dataset_nextday_pm25.csv", 39221),
    "chron_train": ("chronological_train.csv", 27443),
    "chron_val": ("chronological_validation.csv", 5889),
    "chron_test": ("chronological_test.csv", 5889),
    "rand_train": ("random_train.csv", 27454),
    "rand_val": ("random_validation.csv", 5883),
    "rand_test": ("random_test.csv", 5884),
    "dist_train": ("district_holdout_train.csv", 21110),
    "dist_val": ("district_holdout_validation.csv", 4530),
    "dist_test": ("district_holdout_test.csv", 9051),
}

EXPECTED_HELDOUT = ["Dhaka", "Narayanganj", "Narsingdi"]
EXPECTED_TRAIN_DISTRICTS = [

    "Faridpur", "Gazipur", "Gopalganj", "Kishoreganj", "Madaripur",
    "Manikganj", "Munshiganj", "Rajbari", "Shariatpur", "Tangail",
]

EXPECTED_BOUNDARIES = {
    "chron_train_max": "2023-07-09",
    "chron_val_min": "2023-07-10",
    "chron_val_max": "2024-10-04",
    "chron_test_min": "2024-10-05",
    "chron_test_max": "2025-12-31",
}

MODEL_ORDER = ["persistence", "ridge", "random_forest", "hist_gbm"]

# Fixed hyperparameters (identical across comparable experiments; never tuned on test).
HYPERPARAMS = {
    "persistence": {"rule": "prediction = pm25_lag_1d (no fitting)"},
    "ridge": {"imputer": "SimpleImputer(median)", "scaler": "StandardScaler",
              "alpha": 1.0},
    "random_forest": {"imputer": "SimpleImputer(median)", "n_estimators": 200,
                      "max_depth": 15, "min_samples_leaf": 5, "n_jobs": -1},
    "hist_gbm": {"imputer": "SimpleImputer(median)", "max_iter": 200,
                 "learning_rate": 0.05, "max_leaf_nodes": 31,
                 "min_samples_leaf": 20,
                 # Explicitly off: the 'auto' default silently enables early stopping
                 # with an internal random validation split when n > 10,000.
                 "early_stopping": False},
}

PRED_COLUMNS = [
    "experiment_id", "split_type", "seed", "model_name", "partition",
    "target_date", "district", "actual_pm25_ug_m3", "predicted_pm25_ug_m3",

    "residual",
]

log = logging.getLogger("train")


class AuditError(RuntimeError):
    pass


# -------------------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------------------

def setup_logging():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(OUT_DIR / "training.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)


def atomic_write_csv(df, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)

    os.replace(tmp, path)


def atomic_write_json(obj, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_dump_model(obj, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, path)


def zip_models(model_dir, zip_path):
    """Bundle every saved .joblib model into a single zip (atomic write)."""
    model_dir, zip_path = Path(model_dir), Path(zip_path)
    tmp = Path(str(zip_path) + ".tmp")
    files = sorted(model_dir.glob("*.joblib"))
    # .joblib files are already compressed (compress=3), so store without recompressing.
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    os.replace(tmp, zip_path)
    return len(files), zip_path.stat().st_size


def parse_dates(series):
    try:
        parsed = pd.to_datetime(series, format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series)
    return parsed.dt.normalize()


def iso(series):
    return series.dt.strftime("%Y-%m-%d")


def key_set(df):
    return set(zip(df[DIST_COL].astype(str), iso(df[DATE_COL])))



def pkg_version(name):
    try:
        return metadata.version(name)
    except Exception:
        return "unknown"


# -------------------------------------------------------------------------------------
# FEATURE AUDIT
# -------------------------------------------------------------------------------------

def classify_column(col):
    """Return (eligible, category, reason)."""
    if col in METADATA_COLS:
        return False, "metadata", "metadata/target column, never a predictor"
    if HISTORICAL_PATTERN.match(col):
        return True, "historical_lag_or_prior_rolling", \
            "strictly lagged or prior-rolling (uses data before target date)"
    if col in CALENDAR_FEATURES:
        return True, "calendar", "known-in-advance cyclical calendar feature"
    if col.startswith("weather_"):
        return False, "unlagged_weather", \
            "unlagged same-day weather column (target-date leakage risk)"
    return False, "unrecognized", \
        "not a verified lag/prior-rolling/calendar feature (possible same-day variable)"


def run_feature_audit(columns):
    eligible, excluded = [], []
    for col in columns:

        ok, category, reason = classify_column(col)
        if ok:
            eligible.append(col)
        else:
            excluded.append({"column": col, "category": category, "reason": reason})

    json_listed = None
    fc_path = PREP_DIR / "feature_columns.json"
    if fc_path.exists():
        with open(fc_path, "r", encoding="utf-8") as f:
            json_listed = json.load(f)

    unlagged_weather = [e["column"] for e in excluded
                        if e["category"] == "unlagged_weather"]

    audit = {
        "eligible_features": eligible,
        "n_eligible": len(eligible),
        "excluded_columns": excluded,
        "unpatched_preparation_detected": bool(unlagged_weather),
        "unlagged_weather_columns_removed": unlagged_weather,
        "feature_columns_json_listing": json_listed,
        "feature_columns_json_matches_final": (
            json_listed == eligible if json_listed is not None else None),
        "matches_expected_18_set": set(eligible) == set(EXPECTED_FEATURES),
        "matches_expected_18_order": eligible == EXPECTED_FEATURES,
        "rules": {
            "allowed": [
                "pm25_lag_*d / pm25_roll_mean_prev_*d / pm25_roll_std_prev_*d",
                "*_lag_*d / *_roll_mean_prev_*d / *_roll_std_prev_*d (meteorological)",
                "day_of_year/day_of_week/month sin & cos",
            ],

            "rejected": ["unlagged weather_*", "same-day pollutants", "district",
                         "target_date", "target_pm25_ug_m3"],
        },
    }
    return audit


# -------------------------------------------------------------------------------------
# DATA LOADING AND INTEGRITY AUDIT
# -------------------------------------------------------------------------------------

def load_csv(key):
    fname, _ = FILES[key]
    path = PREP_DIR / fname
    if not path.exists() or path.stat().st_size == 0:
        raise AuditError(f"Missing or empty prepared file: {path}")
    df = pd.read_csv(path)
    if DATE_COL not in df.columns:
        raise AuditError(f"{fname}: column '{DATE_COL}' not found")
    df[DATE_COL] = parse_dates(df[DATE_COL])
    df[DIST_COL] = df[DIST_COL].astype(str).str.strip()
    return df


def audit_frame(key, df, features, errors):
    fname, expected_rows = FILES[key]
    info = {"file": fname, "rows": int(len(df))}

    needed = METADATA_COLS + features
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        errors.append(f"{fname}: missing columns {missing_cols}")

        return info

    if len(df) != expected_rows:
        errors.append(f"{fname}: expected {expected_rows} rows, found {len(df)}")

    n_nat = int(df[DATE_COL].isna().sum())
    if n_nat:
        errors.append(f"{fname}: {n_nat} unparseable target_date values")

    n_dup = int(df.duplicated([DIST_COL, DATE_COL]).sum())
    if n_dup:
        errors.append(f"{fname}: {n_dup} duplicate (district, target_date) rows")

    target = pd.to_numeric(df[TARGET_COL], errors="coerce")
    n_nan_target = int(target.isna().sum())
    n_neg_target = int((target < 0).sum())
    if n_nan_target:
        errors.append(f"{fname}: {n_nan_target} missing target values")
    if n_neg_target:
        errors.append(f"{fname}: {n_neg_target} negative target values")

    n_lag1_nan = int(df["pm25_lag_1d"].isna().sum())
    if n_lag1_nan:
        errors.append(f"{fname}: {n_lag1_nan} missing pm25_lag_1d values")

    feat = df[features].apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(feat.to_numpy(dtype="float64")).sum())
    feat = feat.replace([np.inf, -np.inf], np.nan)

    info.update({
        "districts": int(df[DIST_COL].nunique()),
        "date_min": str(df[DATE_COL].min().date()),

        "date_max": str(df[DATE_COL].max().date()),
        "duplicate_keys": n_dup,
        "target_missing": n_nan_target,
        "target_negative": n_neg_target,
        "pm25_lag_1d_missing": n_lag1_nan,
        "infinite_feature_values": n_inf,
        "rows_with_any_feature_nan": int(feat.isna().any(axis=1).sum()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_median": float(target.median()),
    })
    return info


def run_data_audit(frames, features):
    errors, warnings_ = [], []
    summaries = {k: audit_frame(k, df, features, errors) for k, df in frames.items()}

    # Identical schema across files.
    base_cols = list(frames["modeling"].columns)
    for k, df in frames.items():
        if list(df.columns) != base_cols:
            errors.append(f"{FILES[k][0]}: column list/order differs from modeling dataset")

    if errors:
        return summaries, errors, warnings_

    modeling = frames["modeling"]

    # Chronological partition checks.
    ct, cv, cs = frames["chron_train"], frames["chron_val"], frames["chron_test"]
    if not ct[DATE_COL].max() < cv[DATE_COL].min():

        errors.append("chronological: max(train) is not < min(validation)")
    if not cv[DATE_COL].max() < cs[DATE_COL].min():
        errors.append("chronological: max(validation) is not < min(test)")
    if set(ct[DATE_COL]) & set(cv[DATE_COL]) or set(cv[DATE_COL]) & set(cs[DATE_COL]) \
            or set(ct[DATE_COL]) & set(cs[DATE_COL]):
        errors.append("chronological: partitions share calendar dates")
    bounds = {
        "chron_train_max": str(ct[DATE_COL].max().date()),
        "chron_val_min": str(cv[DATE_COL].min().date()),
        "chron_val_max": str(cv[DATE_COL].max().date()),
        "chron_test_min": str(cs[DATE_COL].min().date()),
        "chron_test_max": str(cs[DATE_COL].max().date()),
    }
    for k, v in EXPECTED_BOUNDARIES.items():
        if bounds[k] != v:
            errors.append(f"chronological boundary {k}: expected {v}, found {bounds[k]}")
    if len(ct) + len(cv) + len(cs) != len(modeling) \
            or len(key_set(ct) | key_set(cv) | key_set(cs)) != len(modeling):
        errors.append("chronological partitions do not exactly cover the modeling dataset")

    # Random-split (seed 42) coverage.
    rt, rv, rs = frames["rand_train"], frames["rand_val"], frames["rand_test"]
    if len(key_set(rt) | key_set(rv) | key_set(rs)) != len(modeling) \
            or len(rt) + len(rv) + len(rs) != len(modeling):
        errors.append("random partitions do not exactly cover the modeling dataset")

    # District holdout.
    dt, dv, ds = frames["dist_train"], frames["dist_val"], frames["dist_test"]
    tr_d, va_d, te_d = set(dt[DIST_COL]), set(dv[DIST_COL]), set(ds[DIST_COL])
    if te_d != set(EXPECTED_HELDOUT):
        errors.append(f"district_holdout: test districts {sorted(te_d)} != {EXPECTED_HELDOUT}")
    if tr_d != set(EXPECTED_TRAIN_DISTRICTS):

        errors.append("district_holdout: training districts differ from verified list")
    if va_d != tr_d:
        errors.append("district_holdout: validation districts differ from training districts")
    if te_d & tr_d or te_d & va_d:
        errors.append("district_holdout: test districts overlap training/validation districts")
    if not dt[DATE_COL].max() < dv[DATE_COL].min():
        errors.append("district_holdout: train does not precede validation")
    if str(dt[DATE_COL].max().date()) != EXPECTED_BOUNDARIES["chron_train_max"]:
        errors.append("district_holdout: train end date != 2023-07-09")
    if str(dv[DATE_COL].min().date()) != EXPECTED_BOUNDARIES["chron_val_min"] \
            or str(dv[DATE_COL].max().date()) != EXPECTED_BOUNDARIES["chron_val_max"]:
        errors.append("district_holdout: validation date range != 2023-07-10..2024-10-04")

    per_dist_test = ds.groupby(DIST_COL).size().to_dict()
    if len(set(per_dist_test.values())) != 1:
        warnings_.append(f"district_holdout test rows per district not equal: {per_dist_test}")

    # Cross-check against split_manifest.json when present.
    sm_path = PREP_DIR / "split_manifest.json"
    if sm_path.exists():
        with open(sm_path, "r", encoding="utf-8") as f:
            sm = json.load(f)
        heldout_json = sm.get("district_holdout", {}).get("heldout_districts")
        if heldout_json is not None and sorted(heldout_json) != sorted(EXPECTED_HELDOUT):
            errors.append("split_manifest.json held-out districts differ from verified list")

    return summaries, errors, warnings_


# -------------------------------------------------------------------------------------
# EXPERIMENTS
# -------------------------------------------------------------------------------------


def random_split_indices(n, seed):
    idx = np.arange(n)
    trainval, test = train_test_split(idx, test_size=0.15, random_state=seed, shuffle=True)
    train, val = train_test_split(trainval, test_size=0.15 / 0.85,
                                  random_state=seed, shuffle=True)
    return train, val, test


def build_experiments(frames):
    modeling = frames["modeling"].reset_index(drop=True)
    experiments = []
    split_manifest_files = []

    for seed in RANDOM_SEEDS:
        tr_i, va_i, te_i = random_split_indices(len(modeling), seed)
        exp_id = f"random_seed_{seed}"

        if seed == 42:
            # Verify that the saved random_*.csv files are exactly this two-step split.
            same = (
                key_set(modeling.iloc[tr_i]) == key_set(frames["rand_train"])
                and key_set(modeling.iloc[va_i]) == key_set(frames["rand_val"])
                and key_set(modeling.iloc[te_i]) == key_set(frames["rand_test"])
            )
            if not same:
                raise AuditError(
                    "random_seed_42: saved random_*.csv do not match the reproduced "
                    "train_test_split on the modeling dataset")
            log.info("random_seed_42: saved random_*.csv verified against reproduced split")
            train, val, test = (frames["rand_train"], frames["rand_val"],
                                frames["rand_test"])

        else:
            train = modeling.iloc[tr_i].reset_index(drop=True)
            val = modeling.iloc[va_i].reset_index(drop=True)
            test = modeling.iloc[te_i].reset_index(drop=True)

        manifest = pd.concat([
            pd.DataFrame({"experiment_id": exp_id, "seed": seed, "partition": part,
                          "modeling_row_position": idx,
                          DIST_COL: modeling.loc[idx, DIST_COL].values,
                          DATE_COL: iso(modeling.loc[idx, DATE_COL]).values})
            for part, idx in (("train", tr_i), ("validation", va_i), ("test", te_i))
        ], ignore_index=True)
        mpath = SPLIT_DIR / f"{exp_id}_split.csv"
        atomic_write_csv(manifest, mpath)
        split_manifest_files.append(str(mpath))

        experiments.append({"id": exp_id, "split_type": "random", "seed": seed,
                            "train": train, "val": val, "test": test})

    experiments.append({"id": "chronological", "split_type": "chronological",
                        "seed": BASE_SEED, "train": frames["chron_train"],
                        "val": frames["chron_val"], "test": frames["chron_test"]})
    experiments.append({"id": "district_holdout", "split_type": "district_holdout",
                        "seed": BASE_SEED, "train": frames["dist_train"],
                        "val": frames["dist_val"], "test": frames["dist_test"]})

    for exp in experiments:
        if exp["split_type"] == "random":
            continue
        manifest = pd.concat([
            pd.DataFrame({"experiment_id": exp["id"], "partition": part,
                          DIST_COL: exp[key][DIST_COL].values,

                          DATE_COL: iso(exp[key][DATE_COL]).values})
            for part, key in (("train", "train"), ("validation", "val"), ("test", "test"))
        ], ignore_index=True)
        mpath = SPLIT_DIR / f"{exp['id']}_split.csv"
        atomic_write_csv(manifest, mpath)
        split_manifest_files.append(str(mpath))

    return experiments, split_manifest_files


def split_summary(exp):
    out = {"split_type": exp["split_type"], "seed": exp["seed"]}
    for part in ("train", "val", "test"):
        df = exp[part]
        out[part] = {
            "rows": int(len(df)),
            "districts": int(df[DIST_COL].nunique()),
            "date_min": str(df[DATE_COL].min().date()),
            "date_max": str(df[DATE_COL].max().date()),
            "target_mean": float(df[TARGET_COL].mean()),
            "target_std": float(df[TARGET_COL].std()),
        }
    return out


# -------------------------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------------------------

def build_model(name, seed):
    if name == "ridge":
        return Pipeline([

            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=seed)),
        ])
    if name == "random_forest":
        p = HYPERPARAMS["random_forest"]
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                min_samples_leaf=p["min_samples_leaf"], n_jobs=p["n_jobs"],
                random_state=seed)),
        ])
    if name == "hist_gbm":
        p = HYPERPARAMS["hist_gbm"]
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                max_leaf_nodes=p["max_leaf_nodes"],
                min_samples_leaf=p["min_samples_leaf"],
                early_stopping=p["early_stopping"], random_state=seed)),
        ])
    raise ValueError(f"Unknown model: {name}")


def get_X(df, features):
    return (df[features].apply(pd.to_numeric, errors="coerce")
            .astype("float64").replace([np.inf, -np.inf], np.nan))


def compute_metrics(actual, pred):

    actual = np.asarray(actual, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    return {
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)),
    }


def make_pred_frame(exp, model_name, partition, df, preds):
    preds = np.asarray(preds, dtype="float64")
    out = pd.DataFrame({
        "experiment_id": exp["id"],
        "split_type": exp["split_type"],
        "seed": exp["seed"],
        "model_name": model_name,
        "partition": partition,
        "target_date": iso(df[DATE_COL]).values,
        "district": df[DIST_COL].values,
        "actual_pm25_ug_m3": df[TARGET_COL].to_numpy(dtype="float64"),
        "predicted_pm25_ug_m3": preds,
    })
    out["residual"] = out["actual_pm25_ug_m3"] - out["predicted_pm25_ug_m3"]
    return out[PRED_COLUMNS]


def artifact_paths(exp_id, model_name):
    stem = f"{exp_id}__{model_name}"
    return {
        "model": MODEL_DIR / f"{stem}.joblib",
        "validation": PRED_DIR / f"{stem}__validation.csv",
        "test": PRED_DIR / f"{stem}__test.csv",

    }


def load_valid_predictions(exp, model_name):
    """Return {'validation': df, 'test': df} if saved artifacts are complete and valid."""
    paths = artifact_paths(exp["id"], model_name)
    try:
        for p in paths.values():
            if not p.exists() or p.stat().st_size == 0:
                return None
        result = {}
        for part, ref in (("validation", exp["val"]), ("test", exp["test"])):
            pred = pd.read_csv(paths[part])
            if list(pred.columns) != PRED_COLUMNS:
                return None
            if len(pred) != len(ref):
                return None
            if pred["predicted_pm25_ug_m3"].isna().any() \
                    or not np.isfinite(pred["predicted_pm25_ug_m3"]).all():
                return None
            if (pred["experiment_id"] != exp["id"]).any() \
                    or (pred["model_name"] != model_name).any() \
                    or (pred["partition"] != part).any():
                return None
            if not np.allclose(pred["actual_pm25_ug_m3"].to_numpy(),
                               ref[TARGET_COL].to_numpy(dtype="float64"),
                               rtol=0, atol=1e-9):
                return None
            if (pred["target_date"].to_numpy() != iso(ref[DATE_COL]).to_numpy()).any() \
                    or (pred["district"].to_numpy() != ref[DIST_COL].to_numpy()).any():
                return None
            result[part] = pred

        return result
    except Exception:
        return None


def manifest_row(exp, model_name, preds, runtime, status):
    paths = artifact_paths(exp["id"], model_name)
    val_m = compute_metrics(preds["validation"]["actual_pm25_ug_m3"],
                            preds["validation"]["predicted_pm25_ug_m3"])
    test_m = compute_metrics(preds["test"]["actual_pm25_ug_m3"],
                             preds["test"]["predicted_pm25_ug_m3"])
    hp = dict(HYPERPARAMS[model_name])
    if model_name != "persistence":
        hp["random_state"] = exp["seed"]
    return {
        "experiment_id": exp["id"],
        "split_type": exp["split_type"],
        "seed": exp["seed"],
        "model_name": model_name,
        "status": status,
        "n_train": int(len(exp["train"])),
        "n_validation": int(len(exp["val"])),
        "n_test": int(len(exp["test"])),
        "val_mae": val_m["mae"], "val_rmse": val_m["rmse"], "val_r2": val_m["r2"],
        "test_mae": test_m["mae"], "test_rmse": test_m["rmse"], "test_r2": test_m["r2"],
        "runtime_seconds": runtime,
        "hyperparameters": json.dumps(hp, sort_keys=True),
        "model_path": str(paths["model"]),
        "validation_predictions_path": str(paths["validation"]),
        "test_predictions_path": str(paths["test"]),
    }



def fit_and_save(exp, model_name, features):
    """Fit on the training partition only; predict validation and test."""
    paths = artifact_paths(exp["id"], model_name)
    seed = exp["seed"]
    t0 = time.perf_counter()

    X_val = get_X(exp["val"], features)
    X_test = get_X(exp["test"], features)

    if model_name == "persistence":
        artifact = {"model_type": "persistence", "rule": "prediction = pm25_lag_1d",
                    "feature": "pm25_lag_1d", "fitted": False}
        pred_val = X_val["pm25_lag_1d"].to_numpy()
        pred_test = X_test["pm25_lag_1d"].to_numpy()
    else:
        X_train = get_X(exp["train"], features)
        y_train = exp["train"][TARGET_COL].to_numpy(dtype="float64")
        if np.isnan(y_train).any():
            raise ValueError("Missing training target values")
        model = build_model(model_name, seed)
        model.fit(X_train, y_train)
        artifact = model
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)

    for name, p in (("validation", pred_val), ("test", pred_test)):
        if np.isnan(p).any() or not np.isfinite(p).all():
            raise ValueError(f"Non-finite {name} predictions")

    runtime = time.perf_counter() - t0


    # Model first, then predictions; incomplete pairs are simply re-run later.
    atomic_dump_model(artifact, paths["model"])
    atomic_write_csv(make_pred_frame(exp, model_name, "validation", exp["val"], pred_val),
                     paths["validation"])
    atomic_write_csv(make_pred_frame(exp, model_name, "test", exp["test"], pred_test),
                     paths["test"])
    return runtime


# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    setup_logging()
    for d in (OUT_DIR, MODEL_DIR, PRED_DIR, SPLIT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info("TRAINING PIPELINE STARTED")
    log.info("Python %s | numpy %s | pandas %s | scikit-learn %s | joblib %s",
             platform.python_version(), np.__version__, pd.__version__,
             sklearn.__version__, joblib.__version__)
    log.info("Prepared data directory: %s", PREP_DIR)
    log.info("Output directory: %s", OUT_DIR)

    # ---- 1. Load and audit ----------------------------------------------------------
    if not PREP_DIR.exists():
        raise AuditError(f"Prepared-data directory not found: {PREP_DIR}")

    frames = {key: load_csv(key) for key in FILES}
    for key, df in frames.items():

        log.info("Loaded %-34s rows=%6d cols=%d", FILES[key][0], len(df), df.shape[1])

    feature_audit = run_feature_audit(list(frames["modeling"].columns))
    features = feature_audit["eligible_features"]

    if feature_audit["unpatched_preparation_detected"]:
        log.warning("Unlagged weather columns found and EXCLUDED: %s",
                    feature_audit["unlagged_weather_columns_removed"])
    if not feature_audit["matches_expected_18_set"]:
        atomic_write_json(feature_audit, OUT_DIR / "feature_audit.json")
        raise AuditError(
            f"Eligible features ({len(features)}) differ from the verified 18-feature set. "
            "See feature_audit.json.")
    if not feature_audit["matches_expected_18_order"]:
        log.warning("Eligible feature order differs from the verified order; "
                    "using the CSV column order for every model.")
    atomic_write_json(feature_audit, OUT_DIR / "feature_audit.json")
    log.info("Feature audit passed: %d eligible features, %d excluded columns",
             len(features), len(feature_audit["excluded_columns"]))

    summaries, errors, warnings_ = run_data_audit(frames, features)
    for w in warnings_:
        log.warning("AUDIT WARNING: %s", w)
    atomic_write_json({"file_summaries": summaries, "errors": errors,
                       "warnings": warnings_}, OUT_DIR / "data_audit.json")
    if errors:
        for e in errors:
            log.error("AUDIT FAILURE: %s", e)
        raise AuditError(f"{len(errors)} integrity/leakage audit check(s) failed; "
                         "no models were trained. See data_audit.json.")
    log.info("Data integrity and leakage audit passed (no errors).")


    # ---- 2. Experiments -------------------------------------------------------------
    experiments, split_manifest_files = build_experiments(frames)
    split_summaries = {e["id"]: split_summary(e) for e in experiments}
    for e in experiments:
        s = split_summaries[e["id"]]
        log.info("%-18s train=%6d val=%5d test=%5d | test dates %s..%s | test districts=%d",
                 e["id"], s["train"]["rows"], s["val"]["rows"], s["test"]["rows"],
                 s["test"]["date_min"], s["test"]["date_max"], s["test"]["districts"])

    config = {
        "status": "started",
        "dataset_root": DATASET_ROOT,
        "prepared_data_dir": str(PREP_DIR),
        "output_dir": str(OUT_DIR),
        "base_seed": BASE_SEED,
        "random_split_seeds": RANDOM_SEEDS,
        "features": features,
        "models": MODEL_ORDER,
        "hyperparameters": HYPERPARAMS,
        "split_summaries": split_summaries,
        "split_manifest_files": split_manifest_files,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "matplotlib": pkg_version("matplotlib"),
        },
        "notes": [
            "Target is ESTIMATED PM2.5 derived from CAMS forecasts and ERA5 reanalysis, "
            "not ground-monitor measurements.",

            "district is metadata only and is never a model feature.",
            "All preprocessing is fitted inside Pipelines on the training partition only.",
            "district_holdout test spans all dates for the 3 unseen districts; it "
            "measures spatial, not future-date, generalization.",
        ],
    }
    atomic_write_json(config, OUT_DIR / "training_config.json")

    # ---- 3. Train / resume ----------------------------------------------------------
    manifest_path = OUT_DIR / "completed_experiments_manifest.csv"
    old_runtime = {}
    if manifest_path.exists():
        try:
            old = pd.read_csv(manifest_path)
            for _, r in old.iterrows():
                old_runtime[(r["experiment_id"], r["model_name"])] = r["runtime_seconds"]
        except Exception:
            pass

    manifest_rows, failures = [], []
    for exp in experiments:
        for model_name in MODEL_ORDER:
            tag = f"{exp['id']} / {model_name}"
            existing = load_valid_predictions(exp, model_name)
            if existing is not None:
                log.info("SKIP (valid artifacts found): %s", tag)
                runtime = old_runtime.get((exp["id"], model_name), np.nan)
                manifest_rows.append(
                    manifest_row(exp, model_name, existing, runtime, "reused"))
                atomic_write_csv(pd.DataFrame(manifest_rows), manifest_path)
                continue


            log.info("TRAIN: %s", tag)
            try:
                runtime = fit_and_save(exp, model_name, features)
                preds = load_valid_predictions(exp, model_name)
                if preds is None:
                    raise RuntimeError("Saved artifacts failed post-write validation")
                row = manifest_row(exp, model_name, preds, runtime, "completed")
                manifest_rows.append(row)
                atomic_write_csv(pd.DataFrame(manifest_rows), manifest_path)
                log.info("DONE: %s | val MAE=%.3f RMSE=%.3f R2=%.3f | "
                         "test MAE=%.3f RMSE=%.3f R2=%.3f | %.1fs",
                         tag, row["val_mae"], row["val_rmse"], row["val_r2"],
                         row["test_mae"], row["test_rmse"], row["test_r2"], runtime)
            except Exception as exc:
                failures.append(tag)
                log.error("FAILED: %s | %s", tag, exc)
                log.error(traceback.format_exc())

    # ---- 4. Master prediction table -------------------------------------------------
    master_path = OUT_DIR / "all_predictions.csv"
    parts = []
    for r in manifest_rows:
        parts.append(pd.read_csv(r["validation_predictions_path"]))
        parts.append(pd.read_csv(r["test_predictions_path"]))
    if parts:
        master = pd.concat(parts, ignore_index=True)
        atomic_write_csv(master, master_path)
        log.info("Master predictions table: %s (%d rows)", master_path, len(master))

    config["status"] = "completed" if not failures else "completed_with_failures"
    config["failed_runs"] = failures
    config["n_completed_runs"] = len(manifest_rows)

    config["n_expected_runs"] = len(experiments) * len(MODEL_ORDER)
    atomic_write_json(config, OUT_DIR / "training_config.json")

    # ---- 5. Summary -----------------------------------------------------------------
    if manifest_rows:
        summary = pd.DataFrame(manifest_rows)[
            ["experiment_id", "model_name", "val_mae", "val_rmse", "val_r2",
             "test_mae", "test_rmse", "test_r2", "runtime_seconds", "status"]]
        pd.set_option("display.width", 200)
        print("\n" + "=" * 100)
        print("SUMMARY (validation and test metrics, ug/m3)")
        print("=" * 100)
        print(summary.round(3).to_string(index=False))

    log.info("Runs completed: %d / %d | failures: %d",
             len(manifest_rows), len(experiments) * len(MODEL_ORDER), len(failures))

    # ---- 6. Zip all models for one-click download ----------------------------------
    zip_path = Path("/kaggle/working/all_models.zip")
    n_files, n_bytes = zip_models(MODEL_DIR, zip_path)
    log.info("Zipped %d model files -> %s (%.1f MB)", n_files, zip_path, n_bytes / 1e6)

    log.info("TRAINING PIPELINE FINISHED")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as exc:
        setup_logging()
        log.error("AUDIT ERROR: %s", exc)
        sys.exit(2)
    except Exception as exc:
        setup_logging()
        log.error("UNHANDLED ERROR: %s", exc)
        log.error(traceback.format_exc())
        sys.exit(1)



Cell 2:

!python /kaggle/working/train_models.py