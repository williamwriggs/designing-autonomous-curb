from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/synthetic_av_pudo_events.csv")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df["hour"] = pd.to_numeric(df["hour"], errors="raise").astype(int)

    hourly = df.groupby(["hour", "pullover_type"]).size().reset_index(name="events")
    daily = df.groupby(["day_of_week", "pullover_type"]).size().reset_index(name="events")
    day_hour = df.groupby(["day_of_week", "hour"]).size().reset_index(name="events")

    hourly.to_csv(OUTPUT_DIR / "hourly_events.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "daily_events.csv", index=False)
    day_hour.to_csv(OUTPUT_DIR / "day_hour_events.csv", index=False)

    print(f"Peak hour: {df.groupby('hour').size().idxmax():02d}:00")
    print(f"Wrote temporal summaries to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
