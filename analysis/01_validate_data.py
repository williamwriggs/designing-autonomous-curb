from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/synthetic_av_pudo_events.csv")
REQUIRED_COLUMNS = {
    "pullover_type",
    "day_of_week",
    "hour",
    "adjusted_point",
    "desired_point",
    "curb_geo",
    "polygon_type",
}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if len(df) != 4080:
        raise ValueError(f"Expected 4,080 rows; found {len(df):,}")

    if not set(df["pullover_type"].dropna()).issubset({"Pickup", "Dropoff"}):
        raise ValueError("Unexpected pullover_type value detected")

    hours = pd.to_numeric(df["hour"], errors="coerce")
    if hours.isna().any() or not hours.between(0, 23).all():
        raise ValueError("Hour values must be integers from 0 through 23")

    print("Validation passed")
    print(f"Rows: {len(df):,}")
    print(df["pullover_type"].value_counts())
    print(df.isna().sum())


if __name__ == "__main__":
    main()
