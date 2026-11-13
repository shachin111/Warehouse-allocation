import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from google.colab import files

# ============================================================
# SMART WAREHOUSE INVENTORY MANAGEMENT SYSTEM
# DATASET EXTRACTION, AUDIT AND CLEANING
# ============================================================

print("=" * 65)
print("SMART WAREHOUSE INVENTORY MANAGEMENT SYSTEM")
print("FULL DATA CLEANING PIPELINE")
print("=" * 65)

# ============================================================
# 0. UPLOAD ZIP FILE
# ============================================================

print("\nUPLOAD YOUR WAREHOUSE DATASET ZIP FILE")

uploaded = files.upload()

if not uploaded:
    raise ValueError("No file uploaded!")

ZIP_PATH = Path(next(iter(uploaded.keys())))

RAW_NAME = "intelligent_warehouse_inventory_dataset.csv"

OUT_CSV = Path("cleaned_warehouse_inventory.csv")
OUT_REPORT = Path("cleaning_report.txt")

log = []

def say(msg=""):
    print(msg)
    log.append(str(msg))


# ============================================================
# STEP 1 - EXTRACT
# ============================================================

say("\nSTEP 1 - EXTRACT")
say("-" * 65)

extract_dir = Path("extracted_data")
extract_dir.mkdir(exist_ok=True)

if not zipfile.is_zipfile(ZIP_PATH):
    raise ValueError("Uploaded file is not a valid ZIP file!")

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(extract_dir)
    say(f"ZIP file: {ZIP_PATH.name}")
    say(f"Extracted files: {z.namelist()}")

raw_files = list(extract_dir.rglob(RAW_NAME))

if not raw_files:
    raise FileNotFoundError(
        f"Dataset {RAW_NAME} not found inside ZIP!"
    )

df = pd.read_csv(raw_files[0])

required_cols = [
    "record_id", "timestamp", "current_stock_level",
    "minimum_stock_level", "maximum_stock_capacity",
    "reorder_point", "reorder_quantity", "daily_demand",
    "incoming_quantity", "outgoing_quantity", "order_count",
    "backorder_quantity", "unit_product_cost",
    "inventory_holding_cost", "rfid_tracking_accuracy",
    "shelf_occupancy_pct", "storage_utilization_pct",
    "humidity_pct", "amr_battery_level_pct",
    "route_efficiency_score", "layout_utilization_score",
    "stockout_event", "forecasted_demand", "inventory_status",
    "warehouse_id", "storage_zone", "product_category",
    "inventory_class", "demand_pattern", "order_priority",
    "anomaly_status", "optimization_action"
]

missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

say(f"Dataset loaded successfully: {df.shape}")


# ============================================================
# STEP 2 - INITIAL AUDIT
# ============================================================

say("\nSTEP 2 - INITIAL AUDIT")
say("-" * 65)

rows0, cols0 = df.shape

say(f"Original records: {rows0}")
say(f"Original columns: {cols0}")
say(f"Missing values: {df.isnull().sum().sum()}")
say(f"Duplicate rows: {df.duplicated().sum()}")
say(f"Duplicate record IDs: {df['record_id'].duplicated().sum()}")
say(f"Data types: {df.dtypes.value_counts().to_dict()}")


# ============================================================
# STEP 3 - DATA CLEANING
# ============================================================

say("\nSTEP 3 - DATA CLEANING")
say("-" * 65)


# 3.1 TEXT CLEANING

text_cols = df.select_dtypes(include=["object", "string"]).columns

for c in text_cols:
    df[c] = df[c].astype("string").str.strip()

say(f"3.1 Text cleaning completed: {len(text_cols)} columns")


# 3.2 TIMESTAMP CONVERSION

df["timestamp"] = pd.to_datetime(
    df["timestamp"], errors="coerce"
)

bad_ts = int(df["timestamp"].isna().sum())

df = (
    df.dropna(subset=["timestamp"])
    .sort_values("timestamp")
    .reset_index(drop=True)
)

say(f"3.2 Invalid timestamps removed: {bad_ts}")


# 3.3 DUPLICATE REMOVAL

before = len(df)

df = (
    df.drop_duplicates()
    .drop_duplicates(subset="record_id", keep="first")
    .reset_index(drop=True)
)

say(f"3.3 Duplicate records removed: {before - len(df)}")


# 3.4 MISSING VALUE HANDLING

num_cols = df.select_dtypes(include="number").columns.tolist()

cat_cols = [
    c for c in df.columns
    if c not in num_cols and c != "timestamp"
]

n_missing = int(df.isnull().sum().sum())

for c in num_cols:
    if df[c].isnull().any():
        median_value = df[c].median()
        if pd.notna(median_value):
            df[c] = df[c].fillna(median_value)

for c in cat_cols:
    if df[c].isnull().any():
        mode_value = df[c].mode()
        if not mode_value.empty:
            df[c] = df[c].fillna(mode_value.iloc[0])

say(f"3.4 Missing values handled: {n_missing}")


# 3.5 RANGE VALIDATION

pct_cols = [
    "rfid_tracking_accuracy",
    "shelf_occupancy_pct",
    "storage_utilization_pct",
    "humidity_pct",
    "amr_battery_level_pct",
    "route_efficiency_score",
    "layout_utilization_score"
]

non_neg = [
    "current_stock_level",
    "minimum_stock_level",
    "maximum_stock_capacity",
    "reorder_point",
    "reorder_quantity",
    "daily_demand",
    "incoming_quantity",
    "outgoing_quantity",
    "order_count",
    "backorder_quantity",
    "unit_product_cost",
    "inventory_holding_cost"
]

invalid = pd.Series(False, index=df.index)

for c in pct_cols:
    invalid |= (df[c] < 0) | (df[c] > 100)

for c in non_neg:
    invalid |= df[c] < 0

invalid |= (
    df["current_stock_level"] >
    df["maximum_stock_capacity"]
)

invalid_count = int(invalid.sum())

df = df.loc[~invalid].reset_index(drop=True)

say(f"3.5 Invalid range records removed: {invalid_count}")


# 3.6 BUSINESS RULE VALIDATION

bad_rp = (
    df["reorder_point"] <
    df["minimum_stock_level"]
)

df["reorder_below_min_flag"] = bad_rp.astype("int8")

df.loc[bad_rp, "reorder_point"] = df.loc[
    bad_rp, "minimum_stock_level"
]

say(f"3.6 Reorder point corrections: {int(bad_rp.sum())}")


# 3.7 STOCKOUT ANALYSIS

so_with_stock = int(
    (
        (df["stockout_event"] == 1) &
        (df["current_stock_level"] > 0)
    ).sum()
)

say(f"3.7 Stockout events with positive stock: {so_with_stock}")
say("These records are retained because stockout can indicate unmet demand.")


# 3.8 OUTLIER DETECTION USING IQR

out_report = {}

for c in num_cols:

    if df[c].nunique() <= 2:
        continue

    q1 = df[c].quantile(0.25)
    q3 = df[c].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(
        ((df[c] < lower) | (df[c] > upper)).sum()
    )

    if count > 0:
        out_report[c] = count

top = dict(
    sorted(
        out_report.items(),
        key=lambda x: x[1],
        reverse=True
    )[:6]
)

say(f"3.8 Columns containing IQR outliers: {len(out_report)}")
say(f"Top outlier columns: {top}")
say("Outliers retained for business analysis.")


# 3.9 FEATURE ENGINEERING

df["year"] = df["timestamp"].dt.year.astype("int16")
df["month"] = df["timestamp"].dt.month.astype("int8")
df["day_of_week"] = df["timestamp"].dt.day_name().astype("string")
df["hour"] = df["timestamp"].dt.hour.astype("int8")

df["forecast_error"] = (
    df["forecasted_demand"] -
    df["daily_demand"]
).round(2)

say("3.9 Feature engineering completed")


# 3.10 MEMORY OPTIMIZATION

categorical_cols = [
    "warehouse_id",
    "storage_zone",
    "product_category",
    "inventory_class",
    "demand_pattern",
    "order_priority",
    "anomaly_status",
    "optimization_action",
    "inventory_status",
    "day_of_week"
]

for c in categorical_cols:
    df[c] = df[c].astype("category")

say("3.10 Categorical datatype conversion completed")


# ============================================================
# STEP 4 - FINAL VERIFICATION
# ============================================================

say("\nSTEP 4 - FINAL VERIFICATION")
say("-" * 65)

say(f"Original shape: ({rows0}, {cols0})")
say(f"Final shape: {df.shape}")
say(f"Remaining missing values: {df.isnull().sum().sum()}")
say(f"Remaining duplicate rows: {df.duplicated().sum()}")
say(f"Remaining duplicate record IDs: {df['record_id'].duplicated().sum()}")

say(
    "Inventory status distribution: "
    + str(df["inventory_status"].value_counts().to_dict())
)

say(
    f"Total stockout events: {int(df['stockout_event'].sum())}"
)

say(
    f"Stockout percentage: {df['stockout_event'].mean():.2%}"
)


# ============================================================
# SAVE OUTPUT FILES
# ============================================================

df.to_csv(OUT_CSV, index=False)

OUT_REPORT.write_text(
    "\n".join(log),
    encoding="utf-8"
)

say("\n" + "=" * 65)
say("SMART WAREHOUSE DATA CLEANING COMPLETED")
say("=" * 65)

print("\nOUTPUT FILES GENERATED:")
print("1. cleaned_warehouse_inventory.csv")
print("2. cleaning_report.txt")


# ============================================================
# DOWNLOAD FILES
# ============================================================

files.download(str(OUT_CSV))
files.download(str(OUT_REPORT))

print("\nALL OPERATIONS COMPLETED SUCCESSFULLY!")
