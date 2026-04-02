"""Export a cleaned CSV for Power BI or other tabular tools."""

from __future__ import annotations

from pathlib import Path

from project_core import DEFAULT_TRAIN_TICKERS, build_stock_universe, clean_stock_data

OUTPUT_PATH = Path(__file__).resolve().parent / "cleaned_stock_data.csv"


def export_cleaned_csv() -> Path:
    """Create a cleaned stock dataset after the data-cleaning step."""
    raw_data = build_stock_universe(DEFAULT_TRAIN_TICKERS, lookback_days=365 * 3)
    cleaned_data = clean_stock_data(raw_data).copy()
    cleaned_data["Date"] = cleaned_data["Date"].dt.strftime("%Y-%m-%d")
    cleaned_data.to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


if __name__ == "__main__":
    saved_path = export_cleaned_csv()
    print(f"Saved cleaned dataset to: {saved_path}")
