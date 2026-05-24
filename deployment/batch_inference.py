"""
batch_inference.py
------------------
Batch inference script: reads a CSV, predicts prices, writes output.

Usage:
    python batch_inference.py input.csv output.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "model.pkl"


def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_inference.py <input.csv> <output.csv>")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print(f"Reading input: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Predicting prices for {len(df):,} rows...")
    predictions = model.predict(df)
    df["predicted_price"] = predictions.round(2)

    df.to_csv(output_path, index=False)
    print(f"Predictions written to: {output_path}")
    print(f"Mean predicted price: €{predictions.mean():,.2f}")
    print(f"Median predicted price: €{pd.Series(predictions).median():,.2f}")


if __name__ == "__main__":
    main()
