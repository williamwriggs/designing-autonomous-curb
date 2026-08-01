from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/synthetic_av_pudo_events.csv")
OUTPUT_PATH = Path("outputs/adaptive_curb_allocation.csv")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    hourly = df.groupby("hour").size().reindex(range(24), fill_value=0)
    demand_index = hourly / max(float(hourly.max()), 1.0)

    result = pd.DataFrame({"hour": range(24), "synthetic_events": hourly.values})
    result["av_pudo"] = 10 + 30 * demand_index.values
    result["loading_delivery"] = 10 + 12 * np.exp(-0.5 * ((result["hour"] - 9) / 3.0) ** 2)
    result["transit_accessibility"] = 18.0
    result["parking_storage"] = (
        100
        - result["av_pudo"]
        - result["loading_delivery"]
        - result["transit_accessibility"]
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
