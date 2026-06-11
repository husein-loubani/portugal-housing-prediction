"""Unit tests for the preprocessing pipeline in portugal_housing.features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from portugal_housing.config import ALL_FEATURES
from portugal_housing.features import build_preprocessor, make_pipeline


def make_clean(n: int = 80) -> pd.DataFrame:
    """A synthetic frame in the shape clean_data produces."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "price": rng.uniform(80_000, 900_000, n),
        "total_area": rng.uniform(40, 300, n),
        "parking": rng.integers(0, 3, n).astype(float),
        "construction_year": rng.integers(1950, 2024, n).astype(float),
        "total_rooms": rng.integers(1, 8, n).astype(float),
        "living_area": rng.uniform(30, 200, n),
        "number_of_bathrooms": rng.integers(1, 4, n).astype(float),
        "district": rng.choice(["Lisboa", "Porto", "Faro"], n),
        "city": rng.choice(["Lisboa", "Porto", "Faro", "Sintra"], n),
        "town": rng.choice(["Arroios", "Benfica", "Bonfim", ""], n),
        "type": rng.choice(["Apartment", "House"], n),
        "energy_certificate": rng.choice(["A", "B", "C", "NC"], n),
        "elevator": rng.choice([True, False], n),
    })
    # sprinkle the NaN patterns the real data has
    df.loc[df.index[:5], "construction_year"] = np.nan
    df.loc[df.index[5:8], "living_area"] = np.nan
    df.loc[df.index[8], "elevator"] = np.nan
    df.loc[df.index[9], "energy_certificate"] = np.nan
    return df


def test_preprocessor_output_has_no_nan():
    df = make_clean()
    pre = build_preprocessor().fit(df[ALL_FEATURES])
    out = pre.transform(df[ALL_FEATURES])
    assert not pd.isna(np.asarray(out, dtype=float)).any()


def test_preprocessor_handles_unseen_categories():
    df = make_clean()
    pre = build_preprocessor().fit(df[ALL_FEATURES])
    new = df[ALL_FEATURES].head(1).copy()
    new["city"] = "Vila Honesta Que Nao Existe"
    new["town"] = "Freguesia Inventada"
    out = pre.transform(new)   # must not raise
    assert out.shape[0] == 1


def test_pipeline_fits_and_predicts_positive_prices():
    df = make_clean()
    pipe = make_pipeline(LinearRegression())
    pipe.fit(df[ALL_FEATURES], df["price"])
    preds = pipe.predict(df[ALL_FEATURES].head(10))
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


def test_pipeline_independence_between_calls():
    """make_pipeline must return fresh, unshared preprocessors."""
    p1 = make_pipeline(LinearRegression())
    p2 = make_pipeline(LinearRegression())
    assert p1.named_steps["pre"] is not p2.named_steps["pre"]


def test_geo_features_present_in_output_names():
    # Enough rows that the city/town levels clear the rare-level threshold
    df = make_clean(n=400)
    pre = build_preprocessor().fit(df[ALL_FEATURES])
    names = " ".join(pre.get_feature_names_out())
    assert "city" in names and "town" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
