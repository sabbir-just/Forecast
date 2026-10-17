
# =====================================================================================
# PRE-TRAINING PIPELINE
# Next-Day PM2.5 Forecasting Across Time and Districts in Dhaka Division, Bangladesh
#
# This cell performs:
#   1. Dataset discovery and schema inspection
#   2. Data loading and quality checks
#   3. Daily PM2.5 aggregation where needed
#   4. Missing-date handling (no interpolation)
#   5. Leakage-aware time-series feature engineering
#   6. EDA figures and reports
#   7. Random, chronological, and district-held-out split preparation
#   8. Export of prepared data and experiment manifests
#
# NO MODEL IS TRAINED IN THIS CELL.
# GPU is not required for these steps.
# =====================================================================================

import os
import re
import json
import math
import warnings
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
MIN_VALID_HOURS_PER_DAY = 18
LOCAL_TIMEZONE = "Asia/Dhaka"

# -------------------------------------------------------------------------------------
# 1. PATHS
# -------------------------------------------------------------------------------------

DATASET_ROOT = Path(
    "/kaggle/input/datasets/syedazmulhasansabbir/data-science-project/"
    "A District-Level CAMS Forecast–ERA5 Reanalysis PM-/"
    "dhaka_division_pm_aqi_proxy_dataset/v1.0.0"
)

OUTPUT_DIR = Path("/kaggle/working/pretraining_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Locate the dataset automatically if the supplied path differs slightly.
if not DATASET_ROOT.exists():
    print("Specified path not found. Searching Kaggle input directories...")

    matches = list(
        Path("/kaggle/input").rglob("dhaka_division_pm_aqi_proxy_dataset")
    )

    versioned = [
        p / "v1.0.0"
        for p in matches
        if (p / "v1.0.0").is_dir()
    ]

    if versioned:
        DATASET_ROOT = versioned[0]
    elif matches:
        DATASET_ROOT = matches[0]
    else:
        raise FileNotFoundError(
            "Could not find dhaka_division_pm_aqi_proxy_dataset. "
            "Check the Kaggle dataset attachment and the path."
        )

print("=" * 90)
print("DATASET ROOT:", DATASET_ROOT)
print("OUTPUT DIR:  ", OUTPUT_DIR)
print("=" * 90)

# -------------------------------------------------------------------------------------
# 2. DISCOVER DATA FILES
# -------------------------------------------------------------------------------------

SUPPORTED = (".csv", ".csv.gz", ".parquet", ".xlsx", ".xls")

# Exclude documentation and auxiliary files, not actual observations.
SKIP_NAME_PARTS = (
    "readme", "dictionary", "manifest", "checksum", "metadata",
    "quality_report", "methodology", "provenance", "source_archive",
    "requirements", "notebook", "sha256", "integrity", "license",
    "licence", "description"
)

def is_table_file(path):
    name = path.name.lower()
    return (
        any(name.endswith(ext) for ext in SUPPORTED)
        and not any(token in name for token in SKIP_NAME_PARTS)
    )

all_table_files = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file() and is_table_file(p)
)

if not all_table_files:
    raise FileNotFoundError(
        f"No supported CSV, CSV.GZ, Parquet, or Excel files found under "
        f"{DATASET_ROOT}"
    )

print(f"\nFound {len(all_table_files)} candidate table files.")

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def get_columns(path):
    """Read table schema/header without loading the entire dataset."""
    name = path.name.lower()

    if name.endswith(".parquet"):
        if pq is not None:
            return list(pq.ParquetFile(path).schema.names)
        return list(pd.read_parquet(path).columns)

    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return list(pd.read_csv(path, nrows=0).columns)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return list(pd.read_excel(path, nrows=0).columns)

    return []


def normalise_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


# -------------------------------------------------------------------------------------
# 3. AUTOMATIC COLUMN DETECTION
# -------------------------------------------------------------------------------------

def identify_date_column(columns, filename):
    normalised = {c: normalise_name(c) for c in columns}
    name = normalise_name(filename)

    if "daily" in name:
        priority = [
            "datebst", "localdate", "datebdt", "date",
            "timestampbst", "datetimebst", "validtimebst",
            "timestamplocal", "datetimelocal", "validtimelocal",
            "timestamp", "datetime", "validtime", "time",
            "timestamputc", "datetimeutc", "validtimeutc"
        ]
    else:
        priority = [
            "timestampbst", "datetimebst", "validtimebst", "timebst",
            "timestamplocal", "datetimelocal", "validtimelocal",
            "timestamp", "datetime", "validtime", "time",
            "datebst", "localdate", "date", "timestamputc",
            "datetimeutc", "validtimeutc"
        ]

    for candidate in priority:
        for col, norm in normalised.items():
            if norm == candidate:
                return col

    # Fallback for unconventional names.
    for col, norm in normalised.items():
        if any(x in norm for x in ["timestamp", "datetime", "validtime"]):
            return col

    for col, norm in normalised.items():
        if norm in ("date", "localdate", "datebst", "datebdt"):
            return col

    return None


def identify_district_column(columns):
    normalised = {c: normalise_name(c) for c in columns}

    priority = [
        "districtname", "district", "adm2name", "adm2en",
        "name2", "districtenglishname", "adm2"
    ]

    for candidate in priority:
        for col, norm in normalised.items():
            if norm == candidate:
                return col

    # Fallback, e.g. district_id or a district label column.
    for col, norm in normalised.items():
        if "district" in norm and not any(
            token in norm for token in ["flag", "count", "coverage"]
        ):
            return col

    return None


def identify_pm25_column(columns):
    candidates = []

    excluded = (
        "aqi", "index", "proxy", "nowcast", "complete",
        "coverage", "validhour", "nvalid", "count",
        "flag", "source", "quality", "breakpoint", "ratio"
    )

    for col in columns:
        norm = normalise_name(col)

        has_pm25 = (
            "pm25" in norm
            or "pm2p5" in norm
            or "pm2dot5" in norm
        )

        if not has_pm25:
            continue

        if any(token in norm for token in excluded):
            continue

        score = 0
        if "ugm3" in norm or "ugm" in norm:
            score += 10
        if "concentration" in norm or "surface" in norm:
            score += 4
        if norm in ("pm25", "pm2p5"):
            score += 3
        if "kgm3" in norm:
            score += 2

        candidates.append((score, col))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def map_columns(columns, filename):
    date_col = identify_date_column(columns, filename)
    district_col = identify_district_column(columns)
    pm25_col = identify_pm25_column(columns)

    return {
        "date": date_col,
        "district": district_col,
        "pm25": pm25_col
    }


# Inspect candidate schemas.
inventory = []

for path in all_table_files:
    try:
        columns = get_columns(path)
        mapping = map_columns(columns, path.name)

        score = 0
        if all(mapping.values()):
            score += 100

        fname = path.name.lower()

        if "daily" in fname:
            score += 20
        if "core" in fname:
            score += 6
        if "canonical" in fname:
            score += 5
        if fname.endswith(".parquet"):
            score += 3
        elif fname.endswith(".csv.gz"):
            score += 2

        if "hourly" in fname:
            score -= 6
        if "extended" in fname:
            score -= 2
        if "full" in fname:
            score -= 1

        inventory.append({
            "file": str(path),
            "size_MB": round(path.stat().st_size / (1024 ** 2), 2),
            "columns_count": len(columns),
            "date_column": mapping["date"],
            "district_column": mapping["district"],
            "pm25_column": mapping["pm25"],
            "score": score,
            "columns": " | ".join(map(str, columns))
        })

    except Exception as exc:
        print(f"Could not inspect: {path.name} ({exc})")

inventory_df = pd.DataFrame(inventory)

if inventory_df.empty:
    raise RuntimeError("Could not inspect any candidate table files.")

inventory_df = inventory_df.sort_values(
    ["score", "size_MB"], ascending=[False, True]
).reset_index(drop=True)

inventory_df.to_csv(OUTPUT_DIR / "file_inventory.csv", index=False)

print("\nCandidate tables ranked by schema and filename:")
display(
    inventory_df[
        [
            "file", "size_MB", "columns_count",
            "date_column", "district_column", "pm25_column", "score"
        ]
    ].head(25)
)

valid_candidates = inventory_df[
    inventory_df["date_column"].notna()
    & inventory_df["district_column"].notna()
    & inventory_df["pm25_column"].notna()
]

if valid_candidates.empty:
    print("\nAvailable column names:")
    for _, row in inventory_df.head(15).iterrows():
        print("\nFILE:", row["file"])
        print(row["columns"])

    raise RuntimeError(
        "Could not automatically identify date, district, and PM2.5 "
        "concentration columns. Review the printed schema before proceeding."
    )

selected = valid_candidates.iloc[0]
SOURCE_FILE = Path(selected["file"])

COLUMN_MAP = {
    "date": selected["date_column"],
    "district": selected["district_column"],
    "pm25": selected["pm25_column"]
}

print("\nSelected input file:", SOURCE_FILE)
print("Detected column mapping:", COLUMN_MAP)

# -------------------------------------------------------------------------------------
# 4. LOAD THE SELECTED TABLE
# -------------------------------------------------------------------------------------

def read_table(path):
    name = path.name.lower()

    if name.endswith(".parquet"):
        return pd.read_parquet(path)

    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path, low_memory=False)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


raw = read_table(SOURCE_FILE)

print("\nRaw input shape:", raw.shape)
print("\nFirst five rows:")
display(raw.head())

print("\nDetected data types:")
display(
    raw.dtypes.astype(str)
    .rename_axis("column")
    .reset_index(name="dtype")
)

print("\nMissing values in the first 30 columns:")
display(
    raw.isna().mean()
    .sort_values(ascending=False)
    .head(30)
    .rename("missing_fraction")
    .to_frame()
)

# -------------------------------------------------------------------------------------
# 5. PARSE TIME CORRECTLY
# -------------------------------------------------------------------------------------

def parse_local_time(series, column_name):
    """
    Return naive timestamps representing Bangladesh local time.
    Explicit UTC fields are converted to Asia/Dhaka.
    Naive BST/local timestamps are not shifted.
    """
    norm = normalise_name(column_name)

    if "utc" in norm:
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
        return (
            parsed.dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )

    parsed = pd.to_datetime(series, errors="coerce")

    # Handle timezone-aware or mixed-offset strings.
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return (
            parsed.dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )

    has_explicit_offset = series.astype(str).str.contains(
        r"(Z|[+-]\d{2}:?\d{2})$",
        case=False,
        regex=True
    ).any()

    if has_explicit_offset:
        parsed_utc = pd.to_datetime(series, errors="coerce", utc=True)
        return (
            parsed_utc.dt.tz_convert(LOCAL_TIMEZONE)
            .dt.tz_localize(None)
        )

    return parsed


work = pd.DataFrame()

work["date"] = parse_local_time(
    raw[COLUMN_MAP["date"]],
    COLUMN_MAP["date"]
).dt.normalize()

work["district"] = (
    raw[COLUMN_MAP["district"]]
    .astype("string")
    .str.strip()
)

work["pm25_raw"] = pd.to_numeric(
    raw[COLUMN_MAP["pm25"]],
    errors="coerce"
)

# Convert to micrograms/m^3 only when the source column explicitly indicates kg/m^3.
pm25_norm = normalise_name(COLUMN_MAP["pm25"])
if "kgm3" in pm25_norm:
    print("PM2.5 column indicates kg/m^3; converting to µg/m^3.")
    work["pm25_raw"] = work["pm25_raw"] * 1e9

# Negative concentrations are invalid. Missing values are NOT interpolated.
negative_pm25_count = int((work["pm25_raw"] < 0).sum())
work.loc[work["pm25_raw"] < 0, "pm25_raw"] = np.nan

invalid_date_count = int(work["date"].isna().sum())
invalid_district_count = int(
    (work["district"].isna() | work["district"].eq("")).sum()
)

work = work[
    work["date"].notna()
    & work["district"].notna()
    & work["district"].ne("")
].copy()

print("\nValid rows after date/district checks:", len(work))
print("Rows with invalid dates removed:", invalid_date_count)
print("Rows with missing/empty district removed:", invalid_district_count)
print("Negative PM2.5 values changed to missing:", negative_pm25_count)

# -------------------------------------------------------------------------------------
# 6. IDENTIFY AVAILABLE METEOROLOGICAL FEATURES
# -------------------------------------------------------------------------------------

weather_tokens = (
    "temperature", "temp", "dewpoint", "humidity",
    "windspeed", "u10", "v10", "windu", "windv",
    "precipitation", "precip", "t2m", "d2m",
    "pressure", "blh", "boundarylayer", "radiation", "rain"
)
exclude_weather_tokens = (
    "timestamp", "datetime", "validtime", "date", "time",
    "district", "latitude", "longitude", "winddirection",
    "winddir", "aqi", "pm25", "pm10", "flag", "coverage",
    "quality", "source", "forecast", "lead", "index",
    "count", "breakpoint", "complete", "id"
)

weather_raw_columns = []

for col in raw.columns:
    if col in (
        COLUMN_MAP["date"],
        COLUMN_MAP["district"],
        COLUMN_MAP["pm25"]
    ):
        continue

    norm = normalise_name(col)

    if any(token in norm for token in exclude_weather_tokens):
        continue

    if any(token in norm for token in weather_tokens):
        numeric = pd.to_numeric(raw[col], errors="coerce")

        if (
            numeric.notna().sum() >= max(10, int(0.01 * len(raw)))
            and numeric.nunique(dropna=True) >= 5
        ):
            weather_raw_columns.append(col)

# Limit unnecessary predictors and make experiments more manageable.
weather_raw_columns = weather_raw_columns[:10]

print("\nWeather columns available for historical features:")
print(weather_raw_columns if weather_raw_columns else "None detected")
print("The workflow will still work using past PM2.5 values and calendar features.")

# Add detected numerical weather variables to our working table.
for i, col in enumerate(weather_raw_columns):
    clean_name = re.sub(
        r"[^a-zA-Z0-9]+", "_", str(col)
    ).strip("_").lower()

    out_name = f"weather_{i}_{clean_name}"
    work[out_name] = pd.to_numeric(raw.loc[work.index, col], errors="coerce")

weather_feature_names = {
    col: f"weather_{i}_{re.sub(r'[^a-zA-Z0-9]+', '_', str(col)).strip('_').lower()}"
    for i, col in enumerate(weather_raw_columns)
}

# -------------------------------------------------------------------------------------
# 7. DETERMINE WHETHER INPUT IS HOURLY OR ALREADY DAILY
# -------------------------------------------------------------------------------------

records_per_district_day = work.groupby(
    ["district", "date"]
).size()

median_records_per_day = (
    float(records_per_district_day.median())
    if len(records_per_district_day)
    else 0.0
)

is_hourly = (
    "hourly" in SOURCE_FILE.name.lower()
    or median_records_per_day > 1.5
)

print("\nMedian input records per district/date:", median_records_per_day)
print("Treating source as:", "HOURLY" if is_hourly else "DAILY")

# -------------------------------------------------------------------------------------
# 8. BUILD ONE DAILY RECORD PER DISTRICT
# -------------------------------------------------------------------------------------

aggregation = {
    "pm25_ug_m3": ("pm25_raw", "mean"),
    "pm25_valid_records": ("pm25_raw", "count")
}

for source_col, daily_col in weather_feature_names.items():
    aggregation[daily_col] = (daily_col, "mean")

daily = (
    work.groupby(["district", "date"], as_index=False)
    .agg(**aggregation)
    .sort_values(["district", "date"])
    .reset_index(drop=True)
)

# If input is hourly, require at least 18 valid hourly PM2.5 values per day.
if is_hourly:
    insufficient_coverage = (
        daily["pm25_valid_records"] < MIN_VALID_HOURS_PER_DAY
    )

    print(
        "\nDistrict-days below the minimum valid-hour threshold:",
        int(insufficient_coverage.sum())
    )

    daily.loc[insufficient_coverage, "pm25_ug_m3"] = np.nan

daily.to_csv(OUTPUT_DIR / "daily_pm25_dataset.csv", index=False)

print("\nDaily dataset shape:", daily.shape)
print("Districts:", daily["district"].nunique())
print("Date range:", daily["date"].min(), "to", daily["date"].max())
print(
    "Missing daily PM2.5 fraction:",
    round(daily["pm25_ug_m3"].isna().mean(), 4)
)

print("\nPM2.5 summary (µg/m³):")
display(daily["pm25_ug_m3"].describe().to_frame())

print("\nDaily records per district:")
display(
    daily.groupby("district")
    .agg(
        number_of_days=("date", "nunique"),
        valid_pm25_days=("pm25_ug_m3", "count"),
        mean_pm25=("pm25_ug_m3", "mean"),
        missing_pm25_fraction=("pm25_ug_m3", lambda s: s.isna().mean())
    )
    .sort_values("valid_pm25_days", ascending=False)
)

# -------------------------------------------------------------------------------------
# 9. RESTORE MISSING CALENDAR DAYS
#
# Reindexing creates missing dates but DOES NOT invent/interpolate pollution values.
# This guarantees that lag_1 means the previous calendar day, not the previous row.
# -------------------------------------------------------------------------------------

calendar_frames = []

for district, group in daily.groupby("district", sort=True):
    group = (
        group.sort_values("date")
        .drop_duplicates("date", keep="last")
        .set_index("date")
    )

    full_calendar = pd.date_range(
        start=group.index.min(),
        end=group.index.max(),
        freq="D"
    )

    group = group.reindex(full_calendar)
    group.index.name = "date"
    group["district"] = district

    calendar_frames.append(group.reset_index())

daily_calendar = pd.concat(
    calendar_frames,
    ignore_index=True
).sort_values(["district", "date"]).reset_index(drop=True)

print("\nDaily calendar-expanded shape:", daily_calendar.shape)

# -------------------------------------------------------------------------------------
# 10. LEAKAGE-AWARE FEATURE ENGINEERING
#
# Target at date T = PM2.5 on date T.
# Every PM2.5/weather feature is computed from date T-1 or earlier.
# Calendar features are known in advance.
# -------------------------------------------------------------------------------------

feature_frames = []

for district, group in daily_calendar.groupby("district", sort=True):
    g = group.sort_values("date").copy()

    # PM2.5 lags: only observations strictly before the target date.
    for lag in [1, 2, 3, 7, 14]:
        g[f"pm25_lag_{lag}d"] = g["pm25_ug_m3"].shift(lag)

    # Rolling historical PM2.5 statistics.
    # shift(1) ensures the current target is excluded from the rolling window.
    prior_pm25 = g["pm25_ug_m3"].shift(1)

    for window in [3, 7, 14]:
        g[f"pm25_roll_mean_prev_{window}d"] = (
            prior_pm25.rolling(window, min_periods=window).mean()
        )

    g["pm25_roll_std_prev_7d"] = (
        prior_pm25.rolling(7, min_periods=7).std()
    )

    # Past weather only: previous-day values and rolling historical averages.
    for weather_col in weather_feature_names.values():
        if weather_col not in g.columns:
            continue

        readable_name = weather_col.lower()

        g[f"{readable_name}_lag_1d"] = g[weather_col].shift(1)

        for window in [3, 7]:
            g[f"{readable_name}_roll_mean_prev_{window}d"] = (
                g[weather_col]
                .shift(1)
                .rolling(window, min_periods=window)
                .mean()
            )

    # Calendar information for the date being predicted.
    day_of_year = g["date"].dt.dayofyear
    day_of_week = g["date"].dt.dayofweek
    month = g["date"].dt.month

    g["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    g["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    g["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    g["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    g["month_sin"] = np.sin(2 * np.pi * month / 12)
    g["month_cos"] = np.cos(2 * np.pi * month / 12)

    g["target_date"] = g["date"]
    g["target_pm25_ug_m3"] = g["pm25_ug_m3"]

    feature_frames.append(g)

feature_data = pd.concat(
    feature_frames,
    ignore_index=True
).sort_values(["district", "target_date"]).reset_index(drop=True)

# Build the final feature list. Do not retain same-day raw PM2.5 or weather variables.
metadata_cols = {
    "district", "date", "target_date",
    "pm25_ug_m3", "pm25_valid_records", "target_pm25_ug_m3"
} | set(weather_feature_names.values())

feature_cols = [
    c for c in feature_data.columns
    if c not in metadata_cols
    and (
        c.startswith("pm25_lag_")
        or c.startswith("pm25_roll_")
        or c.endswith("_lag_1d")
        or "_roll_mean_prev_" in c
        or c.startswith("day_of_")
        or c.startswith("month_")
    )
]

# Target must exist and the previous day's PM2.5 must be available.
# Other missing features will be imputed inside the later training pipeline,
# using training data only.
modeling_data = feature_data[
    feature_data["target_pm25_ug_m3"].notna()
    & feature_data["pm25_lag_1d"].notna()
].copy()

modeling_data = modeling_data[
    ["target_date", "district", "target_pm25_ug_m3"] + feature_cols
].copy()

# Replace invalid numerical infinities without filling missing values.
numeric_feature_cols = modeling_data.select_dtypes(
    include=[np.number]
).columns.tolist()

modeling_data[numeric_feature_cols] = modeling_data[
    numeric_feature_cols
].replace([np.inf, -np.inf], np.nan)

if modeling_data.empty:
    raise RuntimeError(
        "No modeling rows were generated. Check the detected date/PM2.5 "
        "columns, units, and data coverage."
    )

if not feature_cols:
    raise RuntimeError("No feature columns were generated.")

modeling_data.to_csv(
    OUTPUT_DIR / "modeling_dataset_nextday_pm25.csv",
    index=False
)

with open(OUTPUT_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\n" + "=" * 90)
print("MODEL-READY DATA SUMMARY (NO TRAINING PERFORMED)")
print("=" * 90)
print("Prepared rows:", len(modeling_data))
print("Districts:", modeling_data["district"].nunique())
print("Target date range:",
      modeling_data["target_date"].min(), "to",
      modeling_data["target_date"].max())
print("Number of features:", len(feature_cols))
print("\nFeature columns:")
for col in feature_cols:
    print(" -", col)

print("\nRemaining feature missingness (top 20):")
display(
    modeling_data[feature_cols]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .head(20)
    .rename("missing_fraction")
    .to_frame()
)

# -------------------------------------------------------------------------------------
# 11. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
#
# The boundaries are based on unique dates, not individual rows.
# All districts on a given date go into the same chronological partition.
# -------------------------------------------------------------------------------------

unique_dates = np.array(
    sorted(pd.to_datetime(modeling_data["target_date"]).unique())
)

if len(unique_dates) < 20:
    raise RuntimeError(
        "Too few unique target dates to produce a reliable chronological split."
    )

train_cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.70) - 1])
validation_cutoff = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.85) - 1])

chron_train = modeling_data[
    modeling_data["target_date"] <= train_cutoff
].copy()

chron_validation = modeling_data[
    (modeling_data["target_date"] > train_cutoff)
    & (modeling_data["target_date"] <= validation_cutoff)
].copy()

chron_test = modeling_data[
    modeling_data["target_date"] > validation_cutoff
].copy()

chronological_splits = {
    "chronological_train.csv": chron_train,
    "chronological_validation.csv": chron_validation,
    "chronological_test.csv": chron_test
}

for filename, frame in chronological_splits.items():
    frame.to_csv(OUTPUT_DIR / filename, index=False)

print("\nChronological split boundaries:")
print("Train through:     ", train_cutoff.date())
print("Validation through:", validation_cutoff.date())
print("Test starts after: ", validation_cutoff.date())

print("\nChronological partition sizes:")
for name, frame in chronological_splits.items():
    print(
        f"{name:32s} rows={len(frame):6d} | "
        f"dates={frame['target_date'].min()} to {frame['target_date'].max()}"
    )

# -------------------------------------------------------------------------------------
# 12. RANDOM SPLIT (REFERENCE COMPARISON ONLY)
#
# This deliberately ignores temporal order and is not the primary
# real-world evaluation.
# -------------------------------------------------------------------------------------

random_trainval, random_test = train_test_split(
    modeling_data,
    test_size=0.15,
    random_state=SEED,
    shuffle=True
)

random_train, random_validation = train_test_split(
    random_trainval,
    test_size=0.15 / 0.85,
    random_state=SEED,
    shuffle=True
)

random_splits = {
    "random_train.csv": random_train,
    "random_validation.csv": random_validation,
    "random_test.csv": random_test
}

for filename, frame in random_splits.items():
    frame.to_csv(OUTPUT_DIR / filename, index=False)

print("\nRandom-split sizes:")
for filename, frame in random_splits.items():
    print(f"{filename:28s} {len(frame):6d} rows")

# -------------------------------------------------------------------------------------
# 13. DISTRICT-HELD-OUT SPLIT
#
# Entire districts are excluded from training and validation.
# The test set contains all available dates for the unseen districts.
# This evaluates spatial generalization separately from future-date testing.
# -------------------------------------------------------------------------------------

district_names = sorted(modeling_data["district"].dropna().unique().tolist())
district_manifest = {}

if len(district_names) >= 5:
    rng = np.random.default_rng(SEED)
    n_holdout = max(2, round(0.20 * len(district_names)))
    n_holdout = min(n_holdout, len(district_names) - 2)

    heldout_districts = sorted(
        rng.choice(district_names, size=n_holdout, replace=False).tolist()
    )

    district_trainpool = modeling_data[
        ~modeling_data["district"].isin(heldout_districts)
    ].copy()

    district_test = modeling_data[
        modeling_data["district"].isin(heldout_districts)
    ].copy()

    pool_dates = np.array(
        sorted(pd.to_datetime(district_trainpool["target_date"]).unique())
    )

    pool_train_cutoff = pd.Timestamp(
        pool_dates[int(len(pool_dates) * 0.70) - 1]
    )
    pool_val_cutoff = pd.Timestamp(
        pool_dates[int(len(pool_dates) * 0.85) - 1]
    )

    district_train = district_trainpool[
        district_trainpool["target_date"] <= pool_train_cutoff
    ].copy()

    district_validation = district_trainpool[
        (district_trainpool["target_date"] > pool_train_cutoff)
        & (district_trainpool["target_date"] <= pool_val_cutoff)
    ].copy()

    district_splits = {
        "district_holdout_train.csv": district_train,
        "district_holdout_validation.csv": district_validation,
        "district_holdout_test.csv": district_test
    }

    for filename, frame in district_splits.items():
        frame.to_csv(OUTPUT_DIR / filename, index=False)

    district_manifest = {
        "heldout_districts": heldout_districts,
        "training_districts": sorted(
            set(district_trainpool["district"].unique())
        ),
        "note": (
            "Held-out test districts are never included in training or "
            "validation. This is a spatial-generalization experiment; "
            "the test dates are not restricted to the future period."
        )
    }

    print("\nHeld-out districts:", heldout_districts)
    print("Training districts:", district_manifest["training_districts"])

else:
    print(
        "\nDistrict-held-out split skipped: fewer than five districts "
        "were detected."
    )

# -------------------------------------------------------------------------------------
# 14. BASIC EDA FIGURES
# -------------------------------------------------------------------------------------

plt.figure(figsize=(12, 5))

daily_mean = (
    daily.groupby("date")["pm25_ug_m3"]
    .mean()
    .sort_index()
)

daily_mean.rolling(30, min_periods=1).mean().plot()

plt.title("30-Day Moving Average of Mean PM2.5 Across Available Districts")
plt.xlabel("Bangladesh local date")
plt.ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_pm25_time_series.png", dpi=200)
plt.show()
plt.close()


plt.figure(figsize=(10, 6))

district_mean = (
    daily.groupby("district")["pm25_ug_m3"]
    .mean()
    .sort_values()
)

district_mean.plot(kind="barh")

plt.title("Mean PM2.5 by District")
plt.xlabel("Mean PM2.5 (µg/m³)")
plt.ylabel("District")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_pm25_by_district.png", dpi=200)
plt.show()
plt.close()


plt.figure(figsize=(11, 5))

missing_fraction = (
    modeling_data[feature_cols]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .head(20)
)

missing_fraction.plot(kind="bar")

plt.title("Feature Missingness Before Training-Time Imputation")
plt.xlabel("Feature")
plt.ylabel("Fraction missing")
plt.xticks(rotation=70, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_feature_missingness.png", dpi=200)
plt.show()
plt.close()


plt.figure(figsize=(10, 5))

modeling_data["target_pm25_ug_m3"].dropna().plot(
    kind="hist", bins=50
)

plt.title("Distribution of Next-Day PM2.5 Targets")
plt.xlabel("Target PM2.5 (µg/m³)")
plt.ylabel("Number of rows")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_target_distribution.png", dpi=200)
plt.show()
plt.close()

# -------------------------------------------------------------------------------------
# 15. SAVE REPORTS AND REPRODUCIBILITY MANIFEST
# -------------------------------------------------------------------------------------

quality_report = {
    "source_file": str(SOURCE_FILE),
    "detected_columns": COLUMN_MAP,
    "source_rows_loaded": int(len(raw)),
    "valid_rows_after_date_and_district_checks": int(len(work)),
    "invalid_dates_removed": int(invalid_date_count),
    "invalid_district_rows_removed": int(invalid_district_count),
    "negative_pm25_values_set_missing": int(negative_pm25_count),
    "source_classified_as": "hourly" if is_hourly else "daily",
    "median_records_per_district_day": median_records_per_day,
    "minimum_valid_hours_if_hourly": MIN_VALID_HOURS_PER_DAY,
    "daily_rows_after_aggregation": int(len(daily)),
    "daily_pm25_missing_fraction": float(daily["pm25_ug_m3"].isna().mean()),
    "modeling_rows": int(len(modeling_data)),
    "district_count": int(modeling_data["district"].nunique()),
    "number_of_features": int(len(feature_cols)),
    "feature_columns": feature_cols,
    "weather_columns_detected": weather_raw_columns,
    "target": "PM2.5 concentration on target_date (µg/m³)",
    "feature_rule": (
        "All PM2.5/weather features use historical lags; "
        "calendar features refer to the date being predicted."
    ),
    "missing_data_rule": (
        "Missing pollution measurements are not interpolated. "
        "Remaining feature missingness must be handled inside model "
        "pipelines fitted on training data only."
    ),
    "chronological_train_last_date": str(train_cutoff.date()),
    "chronological_validation_last_date": str(validation_cutoff.date()),
    "chronological_test_first_date": str(
        pd.Timestamp(chron_test["target_date"].min()).date()
    ),
    "district_holdout": district_manifest
}

with open(OUTPUT_DIR / "data_quality_report.json", "w") as f:
    json.dump(quality_report, f, indent=2, default=str)

split_manifest = {
    "random_split_seed": SEED,
    "random_split_sizes": {
        name: int(len(frame))
        for name, frame in random_splits.items()
    },
    "chronological_split_sizes": {
        name: int(len(frame))
        for name, frame in chronological_splits.items()
    },
    "chronological_train_cutoff": str(train_cutoff.date()),
    "chronological_validation_cutoff": str(validation_cutoff.date()),
    "district_holdout": district_manifest
}

with open(OUTPUT_DIR / "split_manifest.json", "w") as f:
    json.dump(split_manifest, f, indent=2, default=str)

# Bundle outputs for convenient download from Kaggle.
zip_base = "/kaggle/working/pretraining_outputs"
zip_path = shutil.make_archive(
    zip_base,
    "zip",
    root_dir=OUTPUT_DIR
)

print("\n" + "=" * 90)
print("PRE-TRAINING PREPARATION FINISHED")
print("=" * 90)
print("Prepared modeling rows:", len(modeling_data))
print("Features:", len(feature_cols))
print("Output directory:", OUTPUT_DIR)
print("ZIP archive:", zip_path)

print("\nImportant outputs:")
for p in sorted(OUTPUT_DIR.iterdir()):
    print(f" - {p.name}")

print(
    "\nNo models were trained. Review data_quality_report.json, "
    "file_inventory.csv, the plots, and the split sizes before training."
)
