Cell 1:

%%writefile /kaggle/working/revised_experiments.py
# =====================================================================================
# REVISED PM2.5 EXPERIMENTS
# Corrected historical humidity features + leave-one-district-out (LODO)
# Reuses the user's existing .joblib models for original-feature predictions.
# Target: CAMS/ERA5-derived estimated PM2.5, NOT ground-monitor measurements.
# =====================================================================================

import os, re, json, time, sys, traceback, warnings, shutil, logging
from pathlib import Path
from importlib import metadata

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- USER PATHS --------------------------------------------
DATASET_ROOT = Path(
    "/kaggle/input/datasets/syedazmulhasansabbir/data-science-project/"
    "A District-Level CAMS Forecast–ERA5 Reanalysis PM-/"
    "dhaka_division_pm_aqi_proxy_dataset/v1.0.0"
)
PREP_DIR = Path("/kaggle/working/pretraining_outputs")
OLD_MODEL_ROOT = Path("/kaggle/input/models/syedazmulhasansabbir/forecast-models/tensorflow2/default/1")
OUT_DIR = Path("/kaggle/working/revised_experiment_outputs")
MODEL_DIR = OUT_DIR / "models"
PRED_DIR = OUT_DIR / "predictions"
SPLIT_DIR = OUT_DIR / "splits"
for _d in (OUT_DIR, MODEL_DIR, PRED_DIR, SPLIT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
RANDOM_SEEDS = [42, 123, 2025]
TRAIN_END = pd.Timestamp("2023-07-09")
VAL_START = pd.Timestamp("2023-07-10")
VAL_END = pd.Timestamp("2024-10-04")
TEST_START = pd.Timestamp("2024-10-05")
TEST_END = pd.Timestamp("2025-12-31")
LOCAL_TZ = "Asia/Dhaka"
MIN_HOURLY_COVERAGE = 18

DATE = "target_date"
DIST = "district"
TARGET = "target_pm25_ug_m3"
ACT = "actual_pm25_ug_m3"
PRED = "predicted_pm25_ug_m3"

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
HUMIDITY_FEATURES = [
    "relative_humidity_lag_1d",
    "relative_humidity_roll_mean_prev_3d",
    "relative_humidity_roll_mean_prev_7d",
]
FEATURE_SETS = {
    "original_18": BASE_FEATURES,
    "corrected_plus_humidity": BASE_FEATURES + HUMIDITY_FEATURES,
}
MODELS = ["persistence", "ridge", "random_forest", "hist_gbm"]
SPLIT_EXPERIMENTS = ["random_seed_42", "random_seed_123", "random_seed_2025", "chronological", "district_holdout"]
MODEL_PARAMS = {
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5, "n_jobs": -1},
    "hist_gbm": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31,
                 "min_samples_leaf": 20, "early_stopping": False},
}

# ----------------------------- LOGGING / UTILITIES -----------------------------------
log = logging.getLogger("revised_pm25")
log.setLevel(logging.INFO)
log.handlers.clear()
_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
_fh = logging.FileHandler(OUT_DIR / "revised_training.log", mode="a", encoding="utf-8")
_fh.setFormatter(_formatter)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_formatter)
log.addHandler(_fh); log.addHandler(_sh)


def atomic_csv(df, path):
    path = Path(path); tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj, path):
    path = Path(path); tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_joblib(obj, path):
    path = Path(path); tmp = Path(str(path) + ".tmp")
    joblib.dump(obj, tmp, compress=3)
    os.replace(tmp, path)


def normalise_name(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def parse_date_local(series, colname=""):
    """Parse a date column and return naive timestamps expressed as Asia/Dhaka dates/times."""
    n = normalise_name(colname)
    if "utc" in n:
        x = pd.to_datetime(series, errors="coerce", utc=True)
        return x.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    x = pd.to_datetime(series, errors="coerce")
    if isinstance(x.dtype, pd.DatetimeTZDtype):
        return x.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    # If strings explicitly contain an offset, normalize to Dhaka. Naive local/BST timestamps stay local.
    if series.astype(str).str.contains(r"(Z|[+-]\d{2}:?\d{2})$", case=False, regex=True).any():
        x = pd.to_datetime(series, errors="coerce", utc=True)
        return x.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return x


def find_date_col(columns, filename=""):
    norms = {c: normalise_name(c) for c in columns}
    if "daily" in filename.lower():
        priority = ["datebst", "localdate", "datebdt", "date", "timestampbst", "datetimebst", "validtimebst",
                    "timestamplocal", "datetimelocal", "validtimelocal", "timestamp", "datetime", "validtime", "time",
                    "timestamputc", "datetimeutc", "validtimeutc"]
    else:
        priority = ["timestampbst", "datetimebst", "validtimebst", "timebst", "timestamplocal", "datetimelocal",
                    "validtimelocal", "timestamp", "datetime", "validtime", "time", "datebst", "localdate", "date",
                    "timestamputc", "datetimeutc", "validtimeutc"]
    for want in priority:
        for c, n in norms.items():
            if n == want: return c
    for c, n in norms.items():
        if any(t in n for t in ("timestamp", "datetime", "validtime")): return c
    for c, n in norms.items():
        if n in ("date", "localdate", "datebst", "datebdt"): return c
    return None


def find_district_col(columns):
    norms = {c: normalise_name(c) for c in columns}
    priority = ["districtname", "district", "adm2name", "adm2en", "name2", "districtenglishname", "adm2"]
    for want in priority:
        for c, n in norms.items():
            if n == want: return c
    for c, n in norms.items():
        if "district" in n and not any(t in n for t in ("flag", "count", "coverage", "id")):
            return c
    return None


def find_humidity_col(columns):
    candidates = []
    for c in columns:
        original = str(c).lower()
        n = normalise_name(c)
        # Do not use substring 'id' as a reject token: it occurs inside the word humidity.
        if not ("humidity" in n or "relhum" in n or n.startswith("rh")):
            continue
        if any(t in n for t in ("flag", "quality", "coverage", "count", "source", "complete", "validhour", "index", "aqi")):
            continue
        if re.search(r"(^|[^a-z0-9])id([^a-z0-9]|$)", original) or re.search(r"(^|_)id$", original):
            continue
        score = 0
        if "relative" in n or "relhum" in n: score += 5
        if "mean" in n or "avg" in n or "average" in n: score += 4
        if "2m" in n: score += 3
        if "pct" in n or "percent" in n or "percent" in original or "%" in original: score += 2
        if n in ("humidity", "relativehumidity", "rh"): score += 1
        candidates.append((score, c))
    return sorted(candidates, reverse=True, key=lambda x: x[0])[0][1] if candidates else None


def list_table_files(root):
    supported = (".csv", ".csv.gz", ".parquet", ".xlsx", ".xls")
    skip = ("readme", "dictionary", "manifest", "checksum", "metadata", "quality_report", "methodology",
            "provenance", "requirements", "notebook", "sha256", "license", "licence", "description")
    return sorted([p for p in root.rglob("*") if p.is_file()
                   and any(p.name.lower().endswith(x) for x in supported)
                   and not any(s in p.name.lower() for s in skip)])


def table_columns(p):
    n = p.name.lower()
    if n.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(p).schema.names)
        except Exception:
            return list(pd.read_parquet(p).columns)
    if n.endswith(".csv") or n.endswith(".csv.gz"):
        return list(pd.read_csv(p, nrows=0).columns)
    return list(pd.read_excel(p, nrows=0).columns)


def read_table(p):
    n = p.name.lower()
    if n.endswith(".parquet"): return pd.read_parquet(p)
    if n.endswith(".csv") or n.endswith(".csv.gz"): return pd.read_csv(p, low_memory=False)
    return pd.read_excel(p)


def locate_raw_root():
    if DATASET_ROOT.exists(): return DATASET_ROOT
    roots = list(Path("/kaggle/input").rglob("dhaka_division_pm_aqi_proxy_dataset"))
    for r in roots:
        if (r / "v1.0.0").is_dir(): return r / "v1.0.0"
        return r
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")


def read_humidity_daily(raw_root):
    candidates = []
    schema_rows = []
    for p in list_table_files(raw_root):
        try:
            cols = table_columns(p)
            dc, distc, hc = find_date_col(cols, p.name), find_district_col(cols), find_humidity_col(cols)
            schema_rows.append({"file": str(p), "date_col": dc, "district_col": distc, "humidity_col": hc,
                                "size_mb": round(p.stat().st_size / 1024**2, 2)})
            if dc and distc and hc:
                score = 100
                if "daily" in p.name.lower(): score += 50
                if "hourly" in p.name.lower(): score += 10
                if "extended" in p.name.lower(): score -= 5
                if p.name.lower().endswith(".parquet"): score += 2
                candidates.append((score, p, dc, distc, hc))
        except Exception as e:
            log.warning("Could not inspect %s: %s", p.name, e)
    atomic_csv(pd.DataFrame(schema_rows), OUT_DIR / "humidity_source_schema_inventory.csv")
    if not candidates:
        raise RuntimeError("No raw table with date, district, and humidity-like columns was found. "
                           "See humidity_source_schema_inventory.csv and set the correct source/column manually.")
    candidates.sort(key=lambda z: (z[0], -z[1].stat().st_size), reverse=True)
    _, source, dc, distc, hc = candidates[0]
    log.info("Humidity source: %s | date=%s | district=%s | humidity=%s", source, dc, distc, hc)
    raw = read_table(source)
    h = pd.DataFrame({
        "date": parse_date_local(raw[dc], dc).dt.normalize(),
        "district": raw[distc].astype("string").str.strip(),
        "humidity_value": pd.to_numeric(raw[hc], errors="coerce"),
    })
    h = h.dropna(subset=["date", "district"])
    h = h[h["district"].ne("")].copy()
    h.loc[h["humidity_value"] < 0, "humidity_value"] = np.nan
    per_day_counts = h.groupby(["district", "date"]).size()
    hourly_like = ("hourly" in source.name.lower()) or (len(per_day_counts) and per_day_counts.median() > 1.5)
    # When source is hourly, require a minimally covered day. Do not impute missing daily weather.
    daily = h.groupby(["district", "date"], as_index=False).agg(
        relative_humidity=("humidity_value", "mean"), valid_records=("humidity_value", "count"))
    if hourly_like:
        daily.loc[daily["valid_records"] < MIN_HOURLY_COVERAGE, "relative_humidity"] = np.nan
    daily = daily.sort_values(["district", "date"]).drop_duplicates(["district", "date"], keep="last")
    report = {
        "source_file": str(source), "date_column": dc, "district_column": distc, "humidity_column": hc,
        "source_treated_as": "hourly" if hourly_like else "daily",
        "raw_rows": int(len(raw)), "daily_district_date_rows": int(len(daily)),
        "districts": sorted(daily["district"].dropna().unique().tolist()),
        "date_min": str(daily["date"].min()), "date_max": str(daily["date"].max()),
        "valid_daily_humidity_fraction": float(daily["relative_humidity"].notna().mean()),
        "daily_records_requiring_at_least_18_valid_hours": bool(hourly_like),
        "units": "source units retained; inspect the source column documentation before interpreting magnitude"
    }
    atomic_json(report, OUT_DIR / "humidity_source_report.json")
    atomic_csv(daily, OUT_DIR / "daily_humidity_extracted.csv")
    return daily, report


def create_humidity_features(humidity_daily):
    pieces = []
    for district, g in humidity_daily.groupby("district", sort=True):
        g = g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        cal = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(cal)
        g.index.name = DATE
        g[DIST] = district
        h = g["relative_humidity"]
        g[HUMIDITY_FEATURES[0]] = h.shift(1)
        g[HUMIDITY_FEATURES[1]] = h.shift(1).rolling(3, min_periods=3).mean()
        g[HUMIDITY_FEATURES[2]] = h.shift(1).rolling(7, min_periods=7).mean()
        pieces.append(g.reset_index()[[DATE, DIST] + HUMIDITY_FEATURES])
    out = pd.concat(pieces, ignore_index=True)
    out[DATE] = pd.to_datetime(out[DATE]).dt.normalize()
    out[DIST] = out[DIST].astype(str).str.strip()
    return out


def read_frame(path):
    d = pd.read_csv(path, low_memory=False)
    required = [DATE, DIST, TARGET] + BASE_FEATURES
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}. Run the original preparation cell first.")
    d[DATE] = pd.to_datetime(d[DATE], errors="coerce").dt.normalize()
    d[DIST] = d[DIST].astype(str).str.strip()
    for c in BASE_FEATURES + [TARGET]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[DATE, DIST, TARGET, "pm25_lag_1d"]).copy()
    return d


def attach_humidity(frame, hum_features):
    d = frame.copy()
    d[DATE] = pd.to_datetime(d[DATE], errors="coerce").dt.normalize()
    d[DIST] = d[DIST].astype(str).str.strip()
    d = d.merge(hum_features, on=[DATE, DIST], how="left", validate="many_to_one")
    return d


def build_random_split(df, seed):
    idx = np.arange(len(df))
    trainval_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=seed, shuffle=True)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=0.15 / 0.85,
                                          random_state=seed, shuffle=True)
    return {"train": df.iloc[train_idx].copy().reset_index(drop=True),
            "validation": df.iloc[val_idx].copy().reset_index(drop=True),
            "test": df.iloc[test_idx].copy().reset_index(drop=True)}


def model_path_for_source(exp, model):
    exact = OLD_MODEL_ROOT / f"{exp}__{model}.joblib"
    if exact.exists(): return exact
    found = list(OLD_MODEL_ROOT.rglob(f"{exp}__{model}.joblib")) if OLD_MODEL_ROOT.exists() else []
    if found: return found[0]
    return None


def load_old_model(exp, model):
    p = model_path_for_source(exp, model)
    if p is None:
        raise FileNotFoundError(f"Could not find saved model {exp}__{model}.joblib below {OLD_MODEL_ROOT}")
    return joblib.load(p), p


def build_estimator(name, seed):
    if name == "ridge":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    if name == "random_forest":
        p = MODEL_PARAMS[name]
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", RandomForestRegressor(**p, random_state=seed))])
    if name == "hist_gbm":
        return Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("model", HistGradientBoostingRegressor(**MODEL_PARAMS[name], random_state=seed))])
    raise ValueError(name)


def pred_for_model(model_obj, model_name, df, features):
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if model_name == "persistence":
        return X["pm25_lag_1d"].to_numpy(float)
    return np.asarray(model_obj.predict(X), dtype=float)


def prediction_frame(exp, split_type, feature_set, model_name, part, df, prediction):
    out = pd.DataFrame({
        "experiment_id": exp, "split_type": split_type, "feature_set": feature_set,
        "model_name": model_name, "partition": part,
        "target_date": pd.to_datetime(df[DATE]).dt.strftime("%Y-%m-%d").to_numpy(),
        "district": df[DIST].astype(str).to_numpy(),
        ACT: df[TARGET].to_numpy(float), PRED: np.asarray(prediction, float)
    })
    out["residual"] = out[ACT] - out[PRED]
    return out


def load_valid_cached_prediction(path, exp, split_type, feature_set, model_name, part, reference):
    """Reuse a saved prediction file only if metadata, keys, targets, and predictions match."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        cached = pd.read_csv(path)
        expected_cols = ["experiment_id", "split_type", "feature_set", "model_name", "partition",
                         "target_date", "district", ACT, PRED, "residual"]
        if list(cached.columns) != expected_cols or len(cached) != len(reference):
            return None
        if (cached["experiment_id"] != exp).any() or (cached["split_type"] != split_type).any() or \
           (cached["feature_set"] != feature_set).any() or (cached["model_name"] != model_name).any() or \
           (cached["partition"] != part).any():
            return None
        cached["target_date"] = pd.to_datetime(cached["target_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        ref_dates = pd.to_datetime(reference[DATE]).dt.strftime("%Y-%m-%d").to_numpy()
        if (cached["target_date"].to_numpy() != ref_dates).any() or \
           (cached[DIST].astype(str).to_numpy() != reference[DIST].astype(str).to_numpy()).any():
            return None
        if not np.allclose(cached[ACT].to_numpy(float), reference[TARGET].to_numpy(float), rtol=0, atol=1e-8):
            return None
        vals = pd.to_numeric(cached[PRED], errors="coerce").to_numpy(float)
        if not np.isfinite(vals).all():
            return None
        return cached
    except Exception:
        return None


def main():
    t0 = time.time()
    log.info("Starting revised PM2.5 experiments")
    log.info("Model source: %s", OLD_MODEL_ROOT)
    if not OLD_MODEL_ROOT.exists():
        raise FileNotFoundError(f"Saved-model path not found: {OLD_MODEL_ROOT}")
    if not (PREP_DIR / "modeling_dataset_nextday_pm25.csv").exists():
        raise FileNotFoundError(
            f"Missing {PREP_DIR / 'modeling_dataset_nextday_pm25.csv'}. "
            "Run the original pre-training/preparation cell first so the prepared splits exist."
        )

    # 1. Load original prepared modelling data and its exact historical splits.
    base = read_frame(PREP_DIR / "modeling_dataset_nextday_pm25.csv").reset_index(drop=True)
    missing_base_features = [c for c in BASE_FEATURES if c not in base.columns]
    if missing_base_features:
        raise RuntimeError(f"Prepared modelling table lacks original features: {missing_base_features}")
    if base.duplicated([DIST, DATE]).any():
        raise RuntimeError("Duplicate (district, target_date) rows detected in modelling data.")
    log.info("Original modelling rows=%d, districts=%d, date range=%s..%s",
             len(base), base[DIST].nunique(), base[DATE].min().date(), base[DATE].max().date())

    # 2. Recover historical humidity from the raw public dataset.
    raw_root = locate_raw_root()
    humidity_daily, humidity_report = read_humidity_daily(raw_root)
    humidity_features = create_humidity_features(humidity_daily)
    corrected = attach_humidity(base, humidity_features)
    valid_fraction = float(corrected[HUMIDITY_FEATURES[0]].notna().mean())
    if valid_fraction < 0.70:
        raise RuntimeError(f"Only {valid_fraction:.1%} of target rows have a previous-day humidity value. "
                           "Check the humidity source/date alignment before training.")
    feature_audit = {
        "base_feature_count": len(BASE_FEATURES), "base_features": BASE_FEATURES,
        "humidity_features": HUMIDITY_FEATURES,
        "corrected_feature_count": len(BASE_FEATURES) + len(HUMIDITY_FEATURES),
        "humidity_lag1_coverage_on_modelling_rows": valid_fraction,
        "unlagged_weather_predictors_used": False,
        "target_is_estimated_pm25_not_ground_monitor_measurement": True,
        "humidity_source_report": humidity_report,
    }
    atomic_json(feature_audit, OUT_DIR / "revised_feature_audit.json")
    atomic_csv(corrected, OUT_DIR / "modeling_dataset_corrected.csv")
    log.info("Added humidity features. Lag-1 coverage: %.1f%%", 100 * valid_fraction)

    # 3. Assemble the exact original split configurations; attach humidity by district/date.
    exps = {}
    for seed in RANDOM_SEEDS:
        exp = f"random_seed_{seed}"
        if seed == 42:
            files = {part: PREP_DIR / f"random_{part}.csv" for part in ("train", "validation", "test")}
            if not all(p.exists() for p in files.values()):
                exps[exp] = build_random_split(base, seed)
            else:
                raw_parts = {part: read_frame(p) for part, p in files.items()}
                exps[exp] = {part: d.reset_index(drop=True) for part, d in raw_parts.items()}
        else:
            exps[exp] = build_random_split(base, seed)
    for exp, names in {
        "chronological": {"train": "chronological_train.csv", "validation": "chronological_validation.csv", "test": "chronological_test.csv"},
        "district_holdout": {"train": "district_holdout_train.csv", "validation": "district_holdout_validation.csv", "test": "district_holdout_test.csv"},
    }.items():
        split = {}
        for part, fn in names.items():
            p = PREP_DIR / fn
            if not p.exists(): raise FileNotFoundError(f"Required existing split not found: {p}")
            split[part] = read_frame(p).reset_index(drop=True)
        exps[exp] = split

    exps = {exp: {part: attach_humidity(df, humidity_features) for part, df in parts.items()}
            for exp, parts in exps.items()}

    # Save enriched split files for audit/evaluation.
    for exp, parts in exps.items():
        for part, df in parts.items():
            atomic_csv(df, SPLIT_DIR / f"{exp}__{part}.csv")

    # 4. Run previous models as original-18-feature reference; train corrected models.
    all_predictions = []
    run_rows = []
    for ei, exp in enumerate(SPLIT_EXPERIMENTS):
        parts = exps[exp]
        split_type = "random" if exp.startswith("random") else ("chronological" if exp == "chronological" else "district_holdout")
        seed = int(exp.rsplit("_", 1)[1]) if exp.startswith("random_seed_") else SEED
        log.info("Experiment %s | train=%d val=%d test=%d", exp, len(parts["train"]), len(parts["validation"]), len(parts["test"]))
        for model_name in MODELS:
            # Original saved model: no refitting. This makes a genuine reference to earlier results.
            old_obj, old_path = load_old_model(exp, model_name)
            for part in ("validation", "test"):
                p = pred_for_model(old_obj, model_name, parts[part], BASE_FEATURES)
                pf = prediction_frame(exp, split_type, "original_18", model_name, part, parts[part], p)
                all_predictions.append(pf)
                atomic_csv(pf, PRED_DIR / f"{exp}__original_18__{model_name}__{part}.csv")

            # Corrected features: reuse validated cached outputs after interruption where possible.
            model_file = MODEL_DIR / f"{exp}__corrected_plus_humidity__{model_name}.joblib"
            cache_paths = {part: PRED_DIR / f"{exp}__corrected_plus_humidity__{model_name}__{part}.csv"
                           for part in ("validation", "test")}
            cached = {part: load_valid_cached_prediction(cache_paths[part], exp, split_type,
                       "corrected_plus_humidity", model_name, part, parts[part])
                      for part in ("validation", "test")}
            if model_file.exists() and all(cached[p] is not None for p in ("validation", "test")):
                log.info("REUSE valid cached corrected run: %s / %s", exp, model_name)
                all_predictions.extend([cached["validation"], cached["test"]])
                run_rows.append({"experiment_id": exp, "feature_set": "corrected_plus_humidity", "model_name": model_name,
                                 "status": "reused_valid_cached_outputs", "source_model": str(model_file)})
            else:
                if model_name == "persistence":
                    new_obj = {"model_type": "persistence", "feature": "pm25_lag_1d", "fitted": False}
                    atomic_joblib(new_obj, model_file)
                else:
                    Xtr = parts["train"][FEATURE_SETS["corrected_plus_humidity"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                    ytr = parts["train"][TARGET].to_numpy(float)
                    if not np.isfinite(ytr).all(): raise RuntimeError(f"Non-finite training targets: {exp}/{model_name}")
                    new_obj = build_estimator(model_name, seed)
                    new_obj.fit(Xtr, ytr)
                    atomic_joblib(new_obj, model_file)
                for part in ("validation", "test"):
                    p = pred_for_model(new_obj, model_name, parts[part], FEATURE_SETS["corrected_plus_humidity"])
                    pf = prediction_frame(exp, split_type, "corrected_plus_humidity", model_name, part, parts[part], p)
                    all_predictions.append(pf)
                    atomic_csv(pf, cache_paths[part])
                run_rows.append({"experiment_id": exp, "feature_set": "corrected_plus_humidity", "model_name": model_name,
                                 "status": "trained_or_repaired", "source_model": str(model_file)})
            run_rows.append({"experiment_id": exp, "feature_set": "original_18", "model_name": model_name,
                             "status": "reused_existing_model", "source_model": str(old_path)})

    # 5. Leave-one-district-out. Train and test both original and humidity-augmented feature sets.
    districts = sorted(corrected[DIST].dropna().unique().tolist())
    if len(districts) != 13:
        log.warning("Expected 13 districts but found %d: %s", len(districts), districts)
    all_dates = corrected[DATE]
    if not ((all_dates <= TRAIN_END).any() and ((all_dates >= VAL_START) & (all_dates <= VAL_END)).any()
            and ((all_dates >= TEST_START) & (all_dates <= TEST_END)).any()):
        raise RuntimeError("LODO date windows do not overlap with the prepared data range.")

    for fold_i, heldout in enumerate(districts):
        tr_base = base[(base[DATE] <= TRAIN_END) & (base[DIST] != heldout)].copy()
        va_base = base[(base[DATE] >= VAL_START) & (base[DATE] <= VAL_END) & (base[DIST] != heldout)].copy()
        te_base = base[(base[DATE] >= TEST_START) & (base[DATE] <= TEST_END) & (base[DIST] == heldout)].copy()
        tr_corr = corrected[(corrected[DATE] <= TRAIN_END) & (corrected[DIST] != heldout)].copy()
        va_corr = corrected[(corrected[DATE] >= VAL_START) & (corrected[DATE] <= VAL_END) & (corrected[DIST] != heldout)].copy()
        te_corr = corrected[(corrected[DATE] >= TEST_START) & (corrected[DATE] <= TEST_END) & (corrected[DIST] == heldout)].copy()
        if min(len(tr_base), len(va_base), len(te_base)) == 0:
            log.warning("Skipping LODO %s due to empty partition", heldout); continue
        lodo_id = f"lodo_{re.sub(r'[^A-Za-z0-9]+', '_', heldout).strip('_')}"
        log.info("LODO %s | train=%d validation=%d test=%d", heldout, len(tr_base), len(va_base), len(te_base))
        for fs, tr, va, te in (("original_18", tr_base, va_base, te_base),
                               ("corrected_plus_humidity", tr_corr, va_corr, te_corr)):
            feats = FEATURE_SETS[fs]
            for model_name in MODELS:
                mfile = MODEL_DIR / f"{lodo_id}__{fs}__{model_name}.joblib"
                part_frames = {"validation": va, "test": te}
                cache_paths = {part: PRED_DIR / f"{lodo_id}__{fs}__{model_name}__{part}.csv"
                               for part in ("validation", "test")}
                cached = {part: load_valid_cached_prediction(cache_paths[part], lodo_id,
                           "leave_one_district_out", fs, model_name, part, part_frames[part])
                          for part in ("validation", "test")}
                if mfile.exists() and all(cached[p] is not None for p in ("validation", "test")):
                    log.info("REUSE valid cached LODO run: %s / %s / %s", heldout, fs, model_name)
                    all_predictions.extend([cached["validation"], cached["test"]])
                    status = "reused_valid_cached_outputs"
                else:
                    if model_name == "persistence":
                        obj = {"model_type": "persistence", "feature": "pm25_lag_1d", "fitted": False}
                        atomic_joblib(obj, mfile)
                    else:
                        Xtr = tr[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                        ytr = tr[TARGET].to_numpy(float)
                        obj = build_estimator(model_name, SEED + fold_i)
                        obj.fit(Xtr, ytr)
                        atomic_joblib(obj, mfile)
                    for part, frame in part_frames.items():
                        pred = pred_for_model(obj, model_name, frame, feats)
                        pf = prediction_frame(lodo_id, "leave_one_district_out", fs, model_name, part, frame, pred)
                        all_predictions.append(pf)
                        atomic_csv(pf, cache_paths[part])
                    status = "trained_or_repaired_lodo"
                run_rows.append({"experiment_id": lodo_id, "heldout_district": heldout, "feature_set": fs,
                                 "model_name": model_name, "status": status, "source_model": str(mfile),
                                 "n_train": len(tr), "n_validation": len(va), "n_test": len(te)})

    # 6. Save consolidated predictions, manifests, and exact configuration.
    pred_all = pd.concat(all_predictions, ignore_index=True)
    if pred_all[PRED].isna().any() or not np.isfinite(pred_all[PRED].to_numpy(float)).all():
        raise RuntimeError("Non-finite predictions found; see per-experiment files before evaluation.")
    atomic_csv(pred_all, OUT_DIR / "all_revised_predictions.csv")
    atomic_csv(pd.DataFrame(run_rows), OUT_DIR / "revised_runs_manifest.csv")
    config = {
        "dataset_root": str(raw_root), "prepared_directory": str(PREP_DIR), "old_model_root": str(OLD_MODEL_ROOT),
        "output_directory": str(OUT_DIR), "target": "CAMS/ERA5-derived estimated PM2.5; not ground-monitor measurements",
        "base_features": BASE_FEATURES, "humidity_features": HUMIDITY_FEATURES,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()}, "models": MODELS,
        "random_seeds": RANDOM_SEEDS, "chronological_boundaries": {
            "train_end": str(TRAIN_END.date()), "validation_start": str(VAL_START.date()),
            "validation_end": str(VAL_END.date()), "test_start": str(TEST_START.date()), "test_end": str(TEST_END.date())},
        "hyperparameters": MODEL_PARAMS, "python": sys.version, "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        "notes": ["Original-feature baseline uses the existing saved .joblib models.",
                  "Humidity features are strictly lagged or prior-rolling; target-day humidity is excluded.",
                  "LODO models are trained on 12 districts and tested only on the held-out district in the future test period.",
                  "Imputation and scaling are fitted on training partitions only."]
    }
    atomic_json(config, OUT_DIR / "revised_experiment_config.json")
    log.info("Saved %d prediction rows and %d experiment/model records", len(pred_all), len(run_rows))
    log.info("Finished in %.1f min. Outputs: %s", (time.time() - t0) / 60, OUT_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log.error("FAILED: %s", exc)
        log.error(traceback.format_exc())
        raise



Cell 2:

!python /kaggle/working/revised_experiments.py