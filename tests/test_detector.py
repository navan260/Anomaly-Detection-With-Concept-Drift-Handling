"""Unit tests for the Drift-Adaptive Online Anomaly Detection pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detector import (
    NativeADWIN,
    NativeKSWIN,
    NativePageHinkley,
    OnlineAnomalySystem,
    StaticBaselineDetector,
    compute_nab_score,
    evaluate_predictions,
)

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


def test_all_drift_detectors_detect_mean_shift():
    """Verify ADWIN, Page-Hinkley, and KSWIN all detect a sustained regime shift."""
    rng = np.random.default_rng(42)
    seg1 = rng.normal(10.0, 0.25, size=60)
    seg2 = rng.normal(15.0, 0.25, size=60)
    stream = np.concatenate([seg1, seg2])

    for detector in [
        NativeADWIN(delta=0.002, min_window=15, max_window=200),
        NativePageHinkley(delta=0.002, min_window=15, max_window=200),
        NativeKSWIN(delta=0.002, min_window=20, max_window=200, stat_size=15),
    ]:
        fired = any(detector.update(float(v)) for v in stream)
        assert fired, f"{detector.__class__.__name__} failed to detect a +20 sigma regime shift"


def test_nab_score_bounds_and_monotonicity():
    """Verify NAB scoring returns 100 for a perfect detector, 0 for null, and penalizes FPs."""
    y_true = np.zeros(100, dtype=int)
    y_true[40:50] = 1

    null_preds = np.zeros(100, dtype=int)
    perfect_preds = np.zeros(100, dtype=int)
    perfect_preds[40] = 1
    late_preds = np.zeros(100, dtype=int)
    late_preds[49] = 1
    noisy_preds = perfect_preds.copy()
    noisy_preds[[10, 25, 70, 85]] = 1

    s_null = compute_nab_score(null_preds, y_true)
    s_perf = compute_nab_score(perfect_preds, y_true)
    s_late = compute_nab_score(late_preds, y_true)
    s_noisy = compute_nab_score(noisy_preds, y_true)

    assert s_null == 0.0, f"Null detector must score 0.0, got {s_null}"
    assert s_perf == 100.0, f"Earliest-onset detector with 0 FP must score 100.0, got {s_perf}"
    assert 0.0 < s_late < s_perf, f"Late detection ({s_late}) should score less than earliest ({s_perf})"
    assert s_noisy < s_perf, f"Detector with false alarms ({s_noisy}) should score less than clean ({s_perf})"


def test_nab_ambient_adaptive_outperforms_static():
    """Verify adaptive HST+ADWIN beats Static Isolation Forest on real NAB Ambient Temp stream."""
    df = pd.read_csv(DATA_DIR / "nab" / "ambient_temperature_system_failure.csv")
    vals = df["value"].astype(float).to_numpy()
    y_true = df["label"].astype(int).to_numpy()

    adap = OnlineAnomalySystem(warmup=200, k_sigma=2.75, adwin_delta=0.002, adaptive=True)
    adap_preds = np.array([bool(adap.step({"value": float(v)})["alarm"]) for v in vals], dtype=bool)
    m_adap = evaluate_predictions(adap_preds, y_true, model=adap)

    stat = StaticBaselineDetector(method="iforest", warmup=200)
    stat_preds, _ = stat.run_stream(vals)
    m_stat = evaluate_predictions(stat_preds, y_true, model=stat)

    assert m_adap["Faults Caught"] == "2 / 2"
    assert m_adap["False Alarms"] < m_stat["False Alarms"]
    assert m_adap["F1 Score"] > m_stat["F1 Score"]
    assert m_adap["NAB Score"] > m_stat["NAB Score"]
