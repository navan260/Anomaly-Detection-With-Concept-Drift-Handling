"""
Reproducible CLI Benchmark Runner for NAB & Synthetic Drift Streams.
Usage: python scripts/run_benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.detector import (
    OnlineAnomalySystem,
    StaticBaselineDetector,
    evaluate_predictions,
)

STREAMS = [
    ("NAB Ambient Temperature", "ambient_temperature_system_failure.csv", 200, 2.75, 0.002),
    ("NAB Machine Temperature", "machine_temperature_system_failure.csv", 200, 3.5, 0.002),
    ("NAB CPU Utilization", "cpu_utilization_asg_misconfiguration.csv", 200, 3.5, 0.001),
]


def main() -> None:
    cfg = load_config()
    nab_dir = ROOT_DIR / "data" / "nab"

    for title, fname, warmup, k_sigma, delta in STREAMS:
        df = pd.read_csv(nab_dir / fname)
        values = df["value"].astype(float).to_numpy()
        y_true = df["label"].astype(int).to_numpy()

        print(f"\n{'=' * 78}")
        print(f"Stream: {title} ({fname}) | N = {len(values):,} | Warm-up = {warmup} | k = {k_sigma}")
        print(f"{'=' * 78}")

        rows = []
        for name, dm, is_adap in [
            ("HST + ADWIN (Adaptive)", "adwin", True),
            ("HST + Page-Hinkley (Adaptive)", "page_hinkley", True),
            ("HST + KSWIN (Adaptive)", "kswin", True),
            ("Plain HST (No Drift Adapt)", "none", False),
        ]:
            det = OnlineAnomalySystem(
                warmup=warmup,
                k_sigma=k_sigma,
                adwin_delta=delta,
                adaptive=is_adap,
                drift_method=dm,
                cfg=cfg,
            )
            t0 = time.perf_counter()
            preds = np.array([bool(det.step({"value": float(v)})["alarm"]) for v in values], dtype=bool)
            lat = (time.perf_counter() - t0) * 1000.0 / max(len(values), 1)
            m = evaluate_predictions(preds, y_true, lat, det)
            rows.append({"Detector": name, **m})

        for name, method in [
            ("Static Isolation Forest", "iforest"),
            ("Static One-Class SVM", "ocsvm"),
        ]:
            sb = StaticBaselineDetector(method=method, warmup=warmup, cfg=cfg)
            preds, lat = sb.run_stream(values)
            m = evaluate_predictions(preds, y_true, lat, sb)
            rows.append({"Detector": name, **m})

        res_table = pd.DataFrame(rows)
        print(res_table.to_string(index=False))


if __name__ == "__main__":
    main()
