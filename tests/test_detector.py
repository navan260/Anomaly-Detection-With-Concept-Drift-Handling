"""Unit tests for the Drift-Adaptive Online Anomaly Detection pipeline."""
import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import OnlineAnomalySystem, StaticBaselineDetector

DATA_DIR = ROOT_DIR / "data"


def test_normal_stream_has_no_false_alarms():
    """Verify that the clean normal stream raises zero false alarms."""
    df = pd.read_csv(DATA_DIR / "normal.csv")
    detector = OnlineAnomalySystem(warmup=20, k_sigma=3.0)

    alarms = []
    for _, row in df.iterrows():
        out = detector.step({"value": float(row["value"])})
        if out["alarm"]:
            alarms.append(int(row["timestamp"]))

    assert len(alarms) == 0, f"Expected 0 alarms on normal.csv, got alarms at {alarms}"


def test_anomaly_stream_detects_injected_spike():
    """Verify that the injected spike at t=30 in anomaly.csv triggers an alarm."""
    df = pd.read_csv(DATA_DIR / "anomaly.csv")
    detector = OnlineAnomalySystem(warmup=20, k_sigma=3.0)

    alarm_timestamps = []
    for _, row in df.iterrows():
        out = detector.step({"value": float(row["value"])})
        if out["alarm"]:
            alarm_timestamps.append(int(row["timestamp"]))

    assert 30 in alarm_timestamps, f"Expected alarm at t=30, got {alarm_timestamps}"
    assert len(alarm_timestamps) == 1, f"Expected only 1 alarm, got {alarm_timestamps}"


def test_static_vs_adaptive_under_step_drift():
    """Verify that adaptive detector recovers from step drift while static baseline floods."""
    df = pd.read_csv(DATA_DIR / "normal.csv")
    vals = df["value"].astype(float).to_numpy().copy()
    # Inject sudden step shift after t=25
    vals[25:] += 0.85

    adaptive = OnlineAnomalySystem(warmup=20, k_sigma=3.0, adwin_delta=0.005, adaptive=True)
    static_if = StaticBaselineDetector(method="iforest", warmup=20)

    adap_alarms = sum(1 for v in vals if adaptive.step({"value": float(v)})["alarm"])
    static_alarms = sum(1 for v in vals if static_if.step({"value": float(v)})["alarm"])

    assert adap_alarms < static_alarms, (
        f"Adaptive detector ({adap_alarms} alarms) should produce fewer false alarms "
        f"under concept drift than static Isolation Forest ({static_alarms} alarms)."
    )
