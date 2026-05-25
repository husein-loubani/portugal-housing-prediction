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

# Feature groups (after cleaning and renaming).
# Two groups only: numerical and categorical. The boolean `elevator` is a
# binary categorical, so it lives with the other categoricals.
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
    "city",
    "type",
    "energy_certificate",
    "elevator",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

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
# Grids are sized for a 91k-row training set. Each grid has ≤16 combinations
# so GridSearchCV with 5-fold CV completes in minutes rather than hours.
# The most impactful parameters are kept; weaker ones use sensible defaults.

DT_GRID = {
    "clf__max_depth":        [10, 20, None],
    "clf__min_samples_leaf": [1, 10],
}

RF_GRID = {
    "clf__n_estimators":      [200],
    "clf__max_depth":         [20, None],
    "clf__min_samples_split": [2, 5],
    "clf__max_features":      ["sqrt", 0.5],
}

GBM_GRID = {
    "clf__n_estimators":  [200, 400],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth":     [3, 5],
}

XGB_GRID = {
    "clf__n_estimators":     [200, 400],
    "clf__learning_rate":    [0.05, 0.1],
    "clf__max_depth":        [5, 7],
    "clf__subsample":        [0.8],
    "clf__colsample_bytree": [0.8],
}

LGBM_GRID = {
    "clf__n_estimators":      [200, 400],
    "clf__learning_rate":     [0.05, 0.1],
    "clf__num_leaves":        [31, 63],
    "clf__min_child_samples": [10, 20],
}
