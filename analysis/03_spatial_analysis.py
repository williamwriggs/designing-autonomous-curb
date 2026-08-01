from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/synthetic_av_pudo_events.csv")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    polygon = (
        df.groupby(["polygon_type", "pullover_type"])
        .size()
        .reset_index(name="events")
    )
    totals = df.groupby("polygon_type").size().reset_index(name="events")
    totals["share"] = totals["events"] / totals["events"].sum()

    polygon.to_csv(OUTPUT_DIR / "polygon_event_type.csv", index=False)
    totals.to_csv(OUTPUT_DIR / "polygon_totals.csv", index=False)

    print(totals.sort_values("events", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
