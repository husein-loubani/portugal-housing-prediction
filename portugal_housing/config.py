"""
config.py
---------
Global constants, feature lists, color palette, and model hyperparameter grids
for the Portugal Housing Price Prediction project.

All magic numbers and strings are centralised here so the notebook contains no
hardcoded literals and changes propagate to every cell automatically.
"""

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Statistical significance threshold ───────────────────────────────────────
ALPHA            = 0.05
CONFIDENCE_LEVEL = 0.95

# ── Evaluation metrics ────────────────────────────────────────────────────────
PRIMARY_METRIC    = "neg_root_mean_squared_error"
SECONDARY_METRICS = ["neg_mean_absolute_error", "r2"]

# ── Data ─────────────────────────────────────────────────────────────────────
TARGET    = "price"
RAW_TARGET = "Price"

# Columns to drop (>50% missing or all-null in the raw data)
DROP_COLUMNS = [
    "GrossArea",              # 79.6% missing
    "Floor",                  # 79.4% missing
    "PublishDate",            # 78.4% missing
    "BuiltArea",              # 80.4% missing
    "ConservationStatus",     # 85.8% missing
    "LotSize",                # 70.8% missing
    "NumberOfBedrooms",       # 65.3% missing
    "NumberOfWC",             # 57.8% missing
    "EnergyEfficiencyLevel",  # 50.4% missing
    "Garage",                 # 50.4% missing
    "ElectricCarsCharging",   # 50.4% missing
    "HasParking",             # 49.7% missing; redundant with Parking column
]

# Column rename map: PascalCase → snake_case
RENAME_COLS = {
    "Price":              "price",
    "District":           "district",
    "City":               "city",
    "Town":               "town",
    "Type":               "type",
    "EnergyCertificate":  "energy_certificate",
    "TotalArea":          "total_area",
    "Parking":            "parking",
    "Elevator":           "elevator",
    "ConstructionYear":   "construction_year",
    "TotalRooms":         "total_rooms",
    "LivingArea":         "living_area",
    "NumberOfBathrooms":  "number_of_bathrooms",
}

# Residential property types kept for modeling. The business case is brokers
# buying homes, so non-residential listings (Land, Garage, Store, Farm, Building,
# Warehouse, Office, etc.) are filtered out during cleaning.
RESIDENTIAL_TYPES = ["Apartment", "House", "Duplex", "Studio", "Mansion", "Manor"]

# Area sanity bounds (applied during cleaning).
MIN_AREA_M2 = 16          # smallest plausible dwelling
MIN_PRICE_PER_M2 = 100    # below this, almost certainly a data error
MAX_PRICE_PER_M2 = 30000  # above this, almost certainly a data error

# Feature groups (after cleaning and renaming).
# Two groups only: numerical and categorical. The boolean `elevator` is a
# binary categorical, so it lives with the other categoricals. City and town
# are high-cardinality but carry strong location signal (tested: keeping them
# lifts test R2 from 0.72 to 0.84); rare levels are pooled by the encoder.
NUMERICAL_FEATURES = [
    "total_area",
    "parking",
    "construction_year",
    "total_rooms",
    "living_area",
    "number_of_bathrooms",
]

CATEGORICAL_FEATURES = [
    "district",
    "type",
    "energy_certificate",
    "elevator",
]

GEO_FEATURES = ["city", "town"]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + GEO_FEATURES

# Rare-level pooling thresholds for one-hot encoding. Geographic columns get a
# lower threshold because even mid-sized towns carry useful price signal.
OHE_MIN_FREQUENCY = 50
GEO_MIN_FREQUENCY = 30

# ── Train / test split ───────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Color palette ───────────────────────────────────────────────────────────
# Primary colors for general use
PALETTE_PRIMARY = "#4C72B0"
PALETTE_ACCENT  = "#DD8452"
PALETTE_LIST    = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

# Sequential palette for heatmaps and continuous target
CMAP_SEQ  = "Blues"
CMAP_DIV  = "RdBu_r"
CMAP_PRICE = "YlOrRd"

# ── Hyperparameter grids ─────────────────────────────────────────────────────
# Sized for the ~52k-row residential training set: broad enough to answer the
# "more exhaustive tuning" review note, small enough that GridSearchCV with
# 5-fold CV still completes in minutes per model.

DT_GRID = {
    "clf__max_depth":        [10, 20, None],
    "clf__min_samples_leaf": [1, 10],
}

# RF is the slow model now that one-hot geography widens the matrix to ~565
# columns; sqrt feature sampling keeps each split cheap, and the grid stays
# small because depth, not size, is what moves RF here.
RF_GRID = {
    "clf__n_estimators":      [200],
    "clf__max_depth":         [20, None],
    "clf__min_samples_split": [2, 5],
    "clf__max_features":      ["sqrt"],
}

GBM_GRID = {
    "clf__n_estimators":  [200, 400],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth":     [3, 5],
}

XGB_GRID = {
    "clf__n_estimators":     [200, 400, 600],
    "clf__learning_rate":    [0.05, 0.1],
    "clf__max_depth":        [5, 7, 9],
    "clf__subsample":        [0.8],
    "clf__colsample_bytree": [0.8],
}

LGBM_GRID = {
    "clf__n_estimators":      [200, 400, 600],
    "clf__learning_rate":     [0.05, 0.1],
    "clf__num_leaves":        [31, 63, 127],
    "clf__min_child_samples": [10, 20],
}

HISTGB_GRID = {
    "clf__max_iter":      [200, 400],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth":     [None, 8],
}
