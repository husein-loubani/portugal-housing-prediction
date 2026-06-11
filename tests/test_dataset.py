"""Unit tests for the cleaning pipeline in portugal_housing.dataset."""

import numpy as np
import pandas as pd
import pytest

from portugal_housing.config import RESIDENTIAL_TYPES
from portugal_housing.dataset import clean_data


def make_raw(n_normal: int = 60) -> pd.DataFrame:
    """A synthetic raw frame in the shape yfinance... the scraped CSV has."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        "Price": rng.uniform(50_000, 900_000, n_normal),
        "District": "Lisboa",
        "City": "Lisboa",
        "Town": "Arroios",
        "Type": rng.choice(["Apartment", "House"], n_normal),
        "EnergyCertificate": "B",
        "TotalArea": rng.uniform(40, 300, n_normal),
        "Parking": 1.0,
        "Elevator": True,
        "ConstructionYear": 2005.0,
        "TotalRooms": 4.0,
        "LivingArea": rng.uniform(30, 200, n_normal),
        "NumberOfBathrooms": 2.0,
        # columns that clean_data drops for >50% missingness
        "GrossArea": np.nan, "Floor": np.nan, "PublishDate": np.nan,
        "BuiltArea": np.nan, "ConservationStatus": np.nan, "LotSize": np.nan,
        "NumberOfBedrooms": np.nan, "NumberOfWC": np.nan,
        "EnergyEfficiencyLevel": np.nan, "Garage": np.nan,
        "ElectricCarsCharging": np.nan, "HasParking": np.nan,
    })
    base["LivingArea"] = np.minimum(base["LivingArea"], base["TotalArea"] - 1)
    return base


def test_keeps_only_residential_types():
    raw = make_raw()
    raw.loc[raw.index[:5], "Type"] = ["Land", "Garage", "Store", "Farm", "Office"]
    cleaned = clean_data(raw)
    assert set(cleaned["type"]).issubset(set(RESIDENTIAL_TYPES))


def test_drops_non_positive_prices():
    raw = make_raw()
    raw.loc[raw.index[0], "Price"] = -10
    raw.loc[raw.index[1], "Price"] = 0
    cleaned = clean_data(raw)
    assert (cleaned["price"] > 0).all()


def test_negative_area_becomes_nan_not_row_drop():
    raw = make_raw()
    victim_price = 333_333.0
    raw.loc[raw.index[2], "TotalArea"] = -50
    raw.loc[raw.index[2], "Price"] = victim_price
    cleaned = clean_data(raw)
    row = cleaned[cleaned["price"] == victim_price]
    assert len(row) == 1
    assert np.isnan(row["total_area"].iloc[0])


def test_living_area_cannot_exceed_total_area():
    raw = make_raw()
    raw.loc[raw.index[3], "LivingArea"] = raw.loc[raw.index[3], "TotalArea"] + 100
    cleaned = clean_data(raw)
    both = cleaned["total_area"].notna() & cleaned["living_area"].notna()
    assert (cleaned.loc[both, "living_area"] <= cleaned.loc[both, "total_area"]).all()


def test_minimum_dwelling_size_enforced():
    raw = make_raw()
    raw.loc[raw.index[4], "TotalArea"] = 8  # below the 16 m2 floor
    cleaned = clean_data(raw)
    known = cleaned["total_area"].dropna()
    assert (known >= 16).all()


def test_price_per_sqm_bounds():
    raw = make_raw()
    raw.loc[raw.index[5], "Price"] = 100_000
    raw.loc[raw.index[5], "TotalArea"] = 10_000   # 10 EUR/m2: implausible
    cleaned = clean_data(raw)
    ppm2 = cleaned["price"] / cleaned["total_area"]
    known = ppm2.dropna()
    assert ((known >= 100) & (known <= 30_000)).all()


def test_duplicates_removed():
    raw = make_raw()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(raw)
    assert not cleaned.duplicated().any()


def test_columns_renamed_to_snake_case():
    cleaned = clean_data(make_raw())
    expected = {"price", "district", "city", "town", "type", "energy_certificate",
                "total_area", "living_area"}
    assert expected.issubset(set(cleaned.columns))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
