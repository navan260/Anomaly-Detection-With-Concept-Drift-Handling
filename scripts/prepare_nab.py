"""
Convert raw Numenta Anomaly Benchmark (NAB) streams into the project format
`timestamp,value,label`, where label = 1 inside an official NAB anomaly window.

Source: https://github.com/numenta/NAB (data/realKnownCause, labels/combined_windows.json)
Usage:  python scripts/prepare_nab.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "nab_raw"
OUT = ROOT / "data" / "nab"

STREAMS = [
    "ambient_temperature_system_failure.csv",
    "machine_temperature_system_failure.csv",
    "cpu_utilization_asg_misconfiguration.csv",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows = json.loads((RAW / "combined_windows.json").read_text())

    for name in STREAMS:
        df = pd.read_csv(RAW / name, parse_dates=["timestamp"])
        df["label"] = 0
        for start, end in windows.get(f"realKnownCause/{name}", []):
            mask = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
            df.loc[mask, "label"] = 1
        df.to_csv(OUT / name, index=False)
        print(f"{name}: {len(df)} rows, {int(df['label'].sum())} labelled-anomaly rows")


if __name__ == "__main__":
    main()
