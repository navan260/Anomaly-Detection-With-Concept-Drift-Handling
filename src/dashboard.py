"""
Clean, Spacious Dashboard for Drift-Adaptive Streaming Anomaly Detection.
Run with: streamlit run src/dashboard.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.detector as _detector_mod
importlib.reload(_detector_mod)
from src.config import load_config
from src.detector import (
    OnlineAnomalySystem,
    StaticBaselineDetector,
    evaluate_predictions,
)

CFG = load_config()


# ---------------------------------------------------------------------------
# Page Setup — Minimal, Airy Layout (Sidebar Collapsed by Default)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift-Adaptive Anomaly Detection",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 {
            font-family: "Inter", "Segoe UI", system-ui, sans-serif;
            color: #0F172A;
            letter-spacing: -0.02em;
        }
        .app-subtitle {
            color: #64748B;
            font-size: 0.95rem;
            margin-top: -0.5rem;
            margin-bottom: 1.6rem;
        }
        .summary-line {
            background: #F8FAFC;
            border-left: 3px solid #0F172A;
            padding: 0.75rem 1rem;
            border-radius: 0 6px 6px 0;
            color: #334155;
            font-size: 0.92rem;
            margin-top: 0.75rem;
        }
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 0.95rem 1.15rem;
        }
        button[data-baseweb="tab"] {
            font-size: 0.95rem !important;
            padding-left: 1.1rem !important;
            padding-right: 1.1rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Scenarios & Data Loading
# ---------------------------------------------------------------------------
NAB_STREAMS = {
    "NAB: Ambient Temperature (7,267 pts)": ("ambient_temperature_system_failure.csv", "°F"),
    "NAB: Machine Temperature (22,695 pts)": ("machine_temperature_system_failure.csv", "°F"),
    "NAB: CPU Utilization (18,050 pts)": ("cpu_utilization_asg_misconfiguration.csv", "%"),
}

DRIFT_METHOD_LABELS = {
    "ADWIN (Adaptive Windowing)": "adwin",
    "Page-Hinkley (Sequential CUSUM)": "page_hinkley",
    "KSWIN (Kolmogorov-Smirnov)": "kswin",
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    "Sudden Drift + 2 Faults": {
        "desc": "Sudden +3 °C baseline shift at t = 95, with real fault spikes at t = 55 and t = 145.",
        "warmup": 35,
        "k_sigma": 3.5,
        "adwin_delta": 0.002,
        "unit": "°C",
    },
    "Gradual Drift + 2 Faults": {
        "desc": "Slow upward drift (+3.2 °C ramp from t = 70), with real fault spikes at t = 50 and t = 150.",
        "warmup": 35,
        "k_sigma": 3.5,
        "adwin_delta": 0.002,
        "unit": "°C",
    },
    "Recurring Regime Shift": {
        "desc": "Operating level rises to a high-load regime (t = 70..130) and returns to normal, with faults at t = 45 and t = 105.",
        "warmup": 35,
        "k_sigma": 3.5,
        "adwin_delta": 0.002,
        "unit": "°C",
    },
    "Clean Normal Stream": {
        "desc": "Healthy stationary sensor stream with no faults and no drift.",
        "warmup": 35,
        "k_sigma": 3.5,
        "adwin_delta": 0.002,
        "unit": "°C",
    },
    "Real Benchmark: NAB Ambient Temp": {
        "desc": "Real 7,267-point NAB stream with ~10 °F seasonal drift and two labelled system failures.",
        "warmup": 200,
        "k_sigma": 2.75,
        "adwin_delta": 0.002,
        "unit": "°F",
        "nab_file": "ambient_temperature_system_failure.csv",
    },
    "Real Benchmark: NAB Machine Temp": {
        "desc": "Real 22,695-point NAB industrial machine temperature stream with four labelled failure windows.",
        "warmup": 200,
        "k_sigma": 3.5,
        "adwin_delta": 0.002,
        "unit": "°F",
        "nab_file": "machine_temperature_system_failure.csv",
    },
    "Real Benchmark: NAB CPU Utilization": {
        "desc": "Real 18,050-point NAB AWS CPU utilization stream with auto-scaling misconfiguration anomalies.",
        "warmup": 200,
        "k_sigma": 3.5,
        "adwin_delta": 0.001,
        "unit": "%",
        "nab_file": "cpu_utilization_asg_misconfiguration.csv",
    },
    "Custom Dataset / Inject Drift": {
        "desc": "Custom stream configured from the sidebar (click '>' at top-left to adjust).",
        "warmup": 35,
        "k_sigma": 3.0,
        "adwin_delta": 0.002,
        "unit": "Value",
    },
}


def build_preset_stream(scenario: str) -> pd.DataFrame:
    nab_file = SCENARIOS.get(scenario, {}).get("nab_file")
    if nab_file:
        return pd.read_csv(ROOT_DIR / "data" / "nab" / nab_file)

    rng = np.random.default_rng(7)
    n = 180
    t = np.arange(n)
    vals = 24.0 + 0.12 * np.sin(2 * np.pi * t / 30.0) + rng.normal(0.0, 0.22, size=n)
    labels = np.zeros(n, dtype=int)

    if scenario == "Sudden Drift + 2 Faults":
        vals[55] += 4.2
        labels[55] = 1
        vals[95:] += 3.0
        vals[145] += 4.2
        labels[145] = 1
    elif scenario == "Gradual Drift + 2 Faults":
        vals[50] += 4.2
        labels[50] = 1
        vals[70:] += np.linspace(0.0, 3.2, n - 70)
        vals[150] += 4.2
        labels[150] = 1
    elif scenario == "Recurring Regime Shift":
        vals[45] += 4.2
        labels[45] = 1
        vals[70:130] += 3.0
        vals[105] += 4.2
        labels[105] = 1

    return pd.DataFrame({"timestamp": t, "value": np.round(vals, 4), "label": labels})


def load_custom_stream(choice: str, uploaded_file) -> Tuple[pd.DataFrame, str]:
    data_dir = ROOT_DIR / "data"
    unit = "Value"
    if choice in NAB_STREAMS:
        fname, unit = NAB_STREAMS[choice]
        df = pd.read_csv(data_dir / "nab" / fname)
    elif choice == "Upload CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif choice == "Normal Fixture (60 pts)":
        df = pd.read_csv(data_dir / "normal.csv")
    else:
        df = pd.read_csv(data_dir / "anomaly.csv")

    df = df.copy()
    if "value" in df.columns and df["value"].dtype == object:
        df["value"] = df["value"].astype(str).str.split("#").str[0].str.strip().astype(float)
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    if "label" not in df.columns:
        df["label"] = 0
    return df, unit


def apply_injections(
    df: pd.DataFrame,
    drift_type: str,
    drift_start: int,
    drift_magnitude: float,
    inject_spike: bool,
    spike_idx: int,
    spike_magnitude: float,
) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    vals = out["value"].astype(float).to_numpy().copy()
    labels = out["label"].astype(int).to_numpy().copy()

    d_start = min(max(0, drift_start), n - 1)
    if drift_type == "Sudden Step":
        vals[d_start:] += drift_magnitude
    elif drift_type == "Gradual Ramp":
        vals[d_start:] += np.linspace(0.0, drift_magnitude, max(1, n - d_start))
    elif drift_type == "Recurring Toggle":
        mid = d_start + max(5, (n - d_start) // 2)
        vals[d_start:mid] += drift_magnitude

    if inject_spike and 0 <= spike_idx < n:
        vals[spike_idx] += spike_magnitude
        labels[spike_idx] = 1

    out["value"] = vals
    out["label"] = labels
    return out


def contiguous_spans(mask: np.ndarray, steps: np.ndarray) -> List[Tuple[int, int]]:
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(steps[g[0]]), int(steps[g[-1]])) for g in np.split(idx, splits)]


# ---------------------------------------------------------------------------
# Prequential Evaluation & Multi-Detector Benchmark
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Processing sensor stream...")
def run_evaluation(
    df: pd.DataFrame,
    warmup: int,
    k_sigma: float,
    adwin_delta: float,
    drift_method: str,
    cfg: dict,
    _engine_version: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    values = df["value"].astype(float).to_numpy()
    timestamps = df["timestamp"].to_numpy()
    y_true = df["label"].astype(int).to_numpy()

    adaptive_sys = OnlineAnomalySystem(
        warmup=warmup,
        k_sigma=k_sigma,
        adwin_delta=adwin_delta,
        adaptive=True,
        drift_method=drift_method,
        cfg=cfg,
    )
    records: List[dict] = []
    for idx, (val, ts, lbl) in enumerate(zip(values, timestamps, y_true)):
        res = adaptive_sys.step({"value": float(val)}, timestamp=ts)
        records.append(
            {
                "step": idx,
                "timestamp": str(ts),
                "value": float(val),
                "label": int(lbl),
                "score": round(res["score"], 4),
                "hst_score": round(res.get("hst_score", res["score"]), 4),
                "z_score": round(res.get("z_score", 0.0), 4),
                "threshold": round(res["threshold"], 4),
                "baseline_mean": round(res["baseline_mean"], 4),
                "baseline_std": round(res.get("baseline_std", 1.0), 4),
                "alarm": bool(res["alarm"]),
                "drift": bool(res["drift"]),
                "warmup": bool(res["warmup"]),
                "latency_ms": round(res["latency_ms"], 4),
            }
        )
    res_df = pd.DataFrame(records)

    def _run_online_variant(method_name: str, is_adaptive: bool) -> Tuple[np.ndarray, float, OnlineAnomalySystem]:
        if method_name == drift_method and is_adaptive:
            return (
                res_df["alarm"].to_numpy(dtype=bool),
                float(res_df["latency_ms"].mean()),
                adaptive_sys,
            )
        det = OnlineAnomalySystem(
            warmup=warmup,
            k_sigma=k_sigma,
            adwin_delta=adwin_delta,
            adaptive=is_adaptive,
            drift_method=method_name,
            cfg=cfg,
        )
        t0 = time.perf_counter()
        alarms = np.array([bool(det.step({"value": float(v)})["alarm"]) for v in values], dtype=bool)
        lat = (time.perf_counter() - t0) * 1000.0 / max(len(values), 1)
        return alarms, lat, det

    adwin_alarms, adwin_lat, adwin_det = _run_online_variant("adwin", True)
    ph_alarms, ph_lat, ph_det = _run_online_variant("page_hinkley", True)
    ks_alarms, ks_lat, ks_det = _run_online_variant("kswin", True)
    hst_alarms, hst_lat, plain_hst = _run_online_variant("none", False)

    iforest_base = StaticBaselineDetector(method="iforest", warmup=warmup, cfg=cfg)
    ocsvm_base = StaticBaselineDetector(method="ocsvm", warmup=warmup, cfg=cfg)
    if_detail = iforest_base.run_stream_detailed(values)
    oc_alarms, oc_lat = ocsvm_base.run_stream(values)

    res_df["static_if_alarm"] = if_detail["alarms"]
    res_df["static_if_mean"] = round(if_detail["baseline_mean"], 4)

    comparison = {
        "HST + ADWIN (Adaptive)": evaluate_predictions(adwin_alarms, y_true, adwin_lat, adwin_det),
        "HST + Page-Hinkley (Adaptive)": evaluate_predictions(ph_alarms, y_true, ph_lat, ph_det),
        "HST + KSWIN (Adaptive)": evaluate_predictions(ks_alarms, y_true, ks_lat, ks_det),
        "Plain HST (No Drift Adapt)": evaluate_predictions(hst_alarms, y_true, hst_lat, plain_hst),
        "Static Isolation Forest": evaluate_predictions(
            if_detail["alarms"], y_true, if_detail["mean_latency_ms"], iforest_base
        ),
        "Static One-Class SVM": evaluate_predictions(oc_alarms, y_true, oc_lat, ocsvm_base),
    }
    return res_df, comparison


@st.cache_data(show_spinner="Running sensitivity sweep...")
def run_sensitivity_sweep(
    df: pd.DataFrame,
    warmup: int,
    adwin_delta: float,
    drift_method: str,
    cfg: dict,
    _engine_version: int = 8,
) -> pd.DataFrame:
    values = df["value"].astype(float).to_numpy()
    y_true = df["label"].astype(int).to_numpy()
    k_grid = [2.0, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5]
    rows: List[Dict[str, Any]] = []
    for k_val in k_grid:
        det = OnlineAnomalySystem(
            warmup=warmup,
            k_sigma=k_val,
            adwin_delta=adwin_delta,
            adaptive=True,
            drift_method=drift_method,
            cfg=cfg,
        )
        t0 = time.perf_counter()
        preds = np.array([bool(det.step({"value": float(v)})["alarm"]) for v in values], dtype=bool)
        lat = (time.perf_counter() - t0) * 1000.0 / max(len(values), 1)
        m = evaluate_predictions(preds, y_true, lat, det)
        rows.append(
            {
                "k_sigma": k_val,
                "F1 Score": m["F1 Score"],
                "NAB Score": m["NAB Score"],
                "False Alarms": m["False Alarms"],
                "FA / 1k pts": m["FA / 1k pts"],
                "Faults Caught": m["Faults Caught"],
                "Mean Delay (steps)": m["Mean Delay (steps)"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Clean Single-Purpose Chart Builders
# ---------------------------------------------------------------------------
def build_single_stream_chart(view_df: pd.DataFrame, warmup_pts: int, unit: str) -> go.Figure:
    """Spacious, uncluttered chart for Tab 1 — shows the sensor stream, ±2σ adaptive envelope, and alarms."""
    fig = go.Figure()
    if len(view_df) == 0:
        return fig

    steps = view_df["step"].to_numpy()
    post_warmup = view_df[~view_df["warmup"]]

    # Subtle warm-up shading
    w_end = int(steps[min(warmup_pts - 1, len(steps) - 1)])
    fig.add_vrect(
        x0=int(steps[0]),
        x1=w_end,
        fillcolor="#F1F5F9",
        opacity=0.8,
        layer="below",
        line_width=0,
        annotation_text="Warm-up",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#64748B"),
    )

    # Subtle amber shading only on multi-step NAB failure windows
    for s0, s1 in contiguous_spans(view_df["label"].to_numpy() == 1, steps):
        if s1 > s0:
            fig.add_vrect(
                x0=s0,
                x1=s1,
                fillcolor="#FEF3C7",
                opacity=0.65,
                layer="below",
                line_width=0,
            )

    # Adaptive ±2σ normal confidence envelope
    if len(post_warmup) and "baseline_std" in post_warmup.columns:
        upper = post_warmup["baseline_mean"] + 2.0 * post_warmup["baseline_std"]
        lower = post_warmup["baseline_mean"] - 2.0 * post_warmup["baseline_std"]
        fig.add_trace(
            go.Scatter(
                x=post_warmup["step"],
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=post_warmup["step"],
                y=lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.11)",
                line=dict(width=0),
                name="Adaptive ±2σ Band",
                hoverinfo="skip",
            )
        )

    # Sensor stream
    fig.add_trace(
        go.Scatter(
            x=view_df["step"],
            y=view_df["value"],
            mode="lines",
            name=f"Sensor ({unit})",
            line=dict(color="#1E293B", width=2.0),
        )
    )

    # Adaptive baseline
    if len(post_warmup):
        fig.add_trace(
            go.Scatter(
                x=post_warmup["step"],
                y=post_warmup["baseline_mean"],
                mode="lines",
                name="Adaptive Baseline",
                line=dict(color="#10B981", width=2.2, dash="dash"),
            )
        )

    # True faults vs False alarms
    tp_df = view_df[view_df["alarm"] & (view_df["label"] == 1)]
    fp_df = view_df[view_df["alarm"] & (view_df["label"] == 0)]

    if len(tp_df):
        fig.add_trace(
            go.Scatter(
                x=tp_df["step"],
                y=tp_df["value"],
                mode="markers",
                name="True Fault Caught",
                marker=dict(color="#16A34A", size=12, symbol="circle", line=dict(color="#FFFFFF", width=2)),
            )
        )
    if len(fp_df):
        fig.add_trace(
            go.Scatter(
                x=fp_df["step"],
                y=fp_df["value"],
                mode="markers",
                name="False Alarm",
                marker=dict(color="#EF4444", size=9, symbol="x"),
            )
        )

    # Spaced drift markers
    drift_steps = [int(s) for s in view_df.loc[view_df["drift"], "step"]]
    spaced_drifts: List[int] = []
    min_gap = max(15, len(view_df) // 25)
    for s in drift_steps:
        if not spaced_drifts or s - spaced_drifts[-1] > min_gap:
            spaced_drifts.append(s)

    for d_step in spaced_drifts:
        fig.add_vline(x=d_step, line_width=1.5, line_dash="dot", line_color="#0D9488")

    if spaced_drifts:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                name="Drift Adapted",
                line=dict(color="#0D9488", width=2, dash="dot"),
            )
        )

    fig.update_layout(
        height=410,
        margin=dict(l=10, r=10, t=35, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#334155"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hovermode="x unified",
        xaxis=dict(title="Time Step (t)", showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(title=unit, showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    )
    return fig


def build_static_vs_adaptive_chart(view_df: pd.DataFrame, warmup_pts: int, unit: str) -> go.Figure:
    """Clean 2-panel comparison solely contrasting Adaptive vs. Static Baseline."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "Adaptive Detector (Streaming HST + Drift Monitor) — Baseline Tracks Drift",
            "Static Detector (Isolation Forest) — Frozen Baseline Causes False-Alarm Flood",
        ),
    )
    post_warmup = view_df[~view_df["warmup"]]
    w_end = int(view_df["step"].iloc[min(warmup_pts - 1, len(view_df) - 1)])
    fig.add_vrect(x0=0, x1=w_end, fillcolor="#F1F5F9", opacity=0.8, layer="below", line_width=0, row="all", col=1)

    # Top: Adaptive
    fig.add_trace(
        go.Scatter(x=view_df["step"], y=view_df["value"], mode="lines", name="Sensor", line=dict(color="#1E293B", width=1.7)),
        row=1, col=1,
    )
    if len(post_warmup):
        fig.add_trace(
            go.Scatter(x=post_warmup["step"], y=post_warmup["baseline_mean"], mode="lines", name="Adaptive Baseline", line=dict(color="#10B981", width=2, dash="dash")),
            row=1, col=1,
        )
    adap_fp = view_df[view_df["alarm"] & (view_df["label"] == 0)]
    adap_tp = view_df[view_df["alarm"] & (view_df["label"] == 1)]
    if len(adap_tp):
        fig.add_trace(
            go.Scatter(x=adap_tp["step"], y=adap_tp["value"], mode="markers", name="True Fault", marker=dict(color="#16A34A", size=11)),
            row=1, col=1,
        )
    if len(adap_fp):
        fig.add_trace(
            go.Scatter(x=adap_fp["step"], y=adap_fp["value"], mode="markers", name="False Alarm", marker=dict(color="#EF4444", size=8)),
            row=1, col=1,
        )

    # Bottom: Static
    fig.add_trace(
        go.Scatter(x=view_df["step"], y=view_df["value"], mode="lines", showlegend=False, line=dict(color="#475569", width=1.6)),
        row=2, col=1,
    )
    if len(post_warmup):
        fig.add_trace(
            go.Scatter(x=post_warmup["step"], y=post_warmup["static_if_mean"], mode="lines", name="Frozen Baseline", line=dict(color="#94A3B8", width=2, dash="dash")),
            row=2, col=1,
        )
    stat_fp = view_df[view_df["static_if_alarm"] & (view_df["label"] == 0)]
    stat_tp = view_df[view_df["static_if_alarm"] & (view_df["label"] == 1)]
    if len(stat_tp):
        fig.add_trace(
            go.Scatter(x=stat_tp["step"], y=stat_tp["value"], mode="markers", showlegend=False, marker=dict(color="#16A34A", size=11)),
            row=2, col=1,
        )
    if len(stat_fp):
        fig.add_trace(
            go.Scatter(x=stat_fp["step"], y=stat_fp["value"], mode="markers", showlegend=False, marker=dict(color="#EF4444", size=7, opacity=0.8)),
            row=2, col=1,
        )

    fig.update_layout(
        height=490,
        margin=dict(l=10, r=10, t=45, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", row=1, col=1)
    fig.update_xaxes(title_text="Time Step (t)", showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", row=2, col=1)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", row=1, col=1)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0", row=2, col=1)
    return fig


def build_score_chart(view_df: pd.DataFrame, warmup_pts: int, show_components: bool = False) -> go.Figure:
    """Dedicated Anomaly Score vs. Threshold chart for Tab 3 (with optional HST/Z decomposition)."""
    fig = go.Figure()
    post_warmup = view_df[~view_df["warmup"]]
    w_end = int(view_df["step"].iloc[min(warmup_pts - 1, len(view_df) - 1)])
    fig.add_vrect(x0=0, x1=w_end, fillcolor="#F1F5F9", opacity=0.8, layer="below", line_width=0)

    if show_components and "hst_score" in view_df.columns and "z_score" in view_df.columns:
        fig.add_trace(
            go.Scatter(
                x=view_df["step"],
                y=view_df["hst_score"],
                mode="lines",
                name="HST Mass Component",
                line=dict(color="#94A3B8", width=1.2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=view_df["step"],
                y=view_df["z_score"],
                mode="lines",
                name="Z-Deviation Component",
                line=dict(color="#8B5CF6", width=1.2, dash="dot"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=view_df["step"],
            y=view_df["score"],
            mode="lines",
            name="Composite Score (sₜ)",
            line=dict(color="#2563EB", width=1.9),
        )
    )
    if len(post_warmup):
        fig.add_trace(
            go.Scatter(
                x=post_warmup["step"],
                y=post_warmup["threshold"],
                mode="lines",
                name="Adaptive Threshold (μ + k·σ)",
                line=dict(color="#D97706", width=2.1, dash="dash"),
            )
        )
    alarms = view_df[view_df["alarm"]]
    if len(alarms):
        fig.add_trace(
            go.Scatter(
                x=alarms["step"],
                y=alarms["score"],
                mode="markers",
                name="Alarm Triggered",
                marker=dict(color="#DC2626", size=9),
            )
        )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=30, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hovermode="x unified",
        xaxis=dict(title="Time Step (t)", showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
        yaxis=dict(title="Score [0–1]", range=[0, 1.05], showgrid=True, gridcolor="#F1F5F9", linecolor="#E2E8F0"),
    )
    return fig


def build_sensitivity_chart(sweep_df: pd.DataFrame) -> go.Figure:
    """Dual-axis sensitivity curve showing F1 / NAB score vs. False Alarms across k_sigma."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=sweep_df["k_sigma"],
            y=sweep_df["F1 Score"],
            mode="lines+markers",
            name="Window-Adjusted F1",
            line=dict(color="#16A34A", width=2.4),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep_df["k_sigma"],
            y=sweep_df["NAB Score"] / 100.0,
            mode="lines+markers",
            name="NAB Score (÷100)",
            line=dict(color="#2563EB", width=2.0, dash="dash"),
            marker=dict(size=7),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=sweep_df["k_sigma"],
            y=sweep_df["False Alarms"],
            name="False Alarms",
            marker_color="rgba(239, 68, 68, 0.28)",
            width=0.16,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=15),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        hovermode="x unified",
        xaxis=dict(title="Threshold Multiplier (k_sigma)", showgrid=True, gridcolor="#F1F5F9"),
    )
    fig.update_yaxes(title_text="F1 / Normalized NAB [0–1]", range=[0, 1.05], showgrid=True, gridcolor="#F1F5F9", secondary_y=False)
    fig.update_yaxes(title_text="False Alarms (count)", showgrid=False, secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# Clean Header + Single Control Row
# ---------------------------------------------------------------------------
st.markdown("## Drift-Adaptive Streaming Anomaly Detection")
st.markdown(
    '<div class="app-subtitle">Online Half-Space Trees + Drift Adaptation (ADWIN / Page-Hinkley / KSWIN) &nbsp;•&nbsp; AIML Lab (IS353IA), RVCE</div>',
    unsafe_allow_html=True,
)

top_c1, top_c2 = st.columns([3.5, 1.0])
with top_c1:
    selected_scenario = st.selectbox(
        "Scenario",
        options=list(SCENARIOS.keys()),
        index=0,
        label_visibility="collapsed",
    )
with top_c2:
    play_clicked = st.button("▶ Replay Live", width="stretch")

preset = SCENARIOS[selected_scenario]


# ---------------------------------------------------------------------------
# Tucked-Away Sidebar (Advanced Parameters & Custom Dataset Only)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Advanced Settings")
    drift_label = st.selectbox(
        "Drift Monitor Algorithm",
        options=list(DRIFT_METHOD_LABELS.keys()),
        index=0,
    )
    drift_method = DRIFT_METHOD_LABELS[drift_label]

    if selected_scenario == "Custom Dataset / Inject Drift":
        custom_choice = st.selectbox(
            "Dataset",
            [
                *NAB_STREAMS.keys(),
                "Anomaly Fixture (60 pts)",
                "Normal Fixture (60 pts)",
                "Upload CSV",
            ],
        )
        uploaded = st.file_uploader("CSV File", type=["csv"]) if custom_choice == "Upload CSV" else None
        base_df, unit = load_custom_stream(custom_choice, uploaded)
    else:
        base_df = build_preset_stream(selected_scenario)
        unit = preset["unit"]

    if len(base_df) > 1000:
        max_pts = st.slider("Samples to Replay", 1000, len(base_df), len(base_df), step=500)
        base_df = base_df.head(max_pts).reset_index(drop=True)

    n_samples = len(base_df)
    value_std = max(float(base_df["value"].std()), 0.05)

    min_warmup = int(CFG["warmup"]["min_samples"])
    max_warmup = max(min_warmup + 5, n_samples // 2)
    raw_warmup = (
        int(CFG["warmup"]["samples_long_stream"])
        if selected_scenario == "Custom Dataset / Inject Drift" and n_samples > 1000
        else int(preset["warmup"])
    )
    warmup_pts = st.slider(
        "Warm-up Samples (N₀)",
        min_warmup,
        max_warmup,
        min(max(raw_warmup, min_warmup), max_warmup),
        key=f"w_{selected_scenario}",
    )
    k_sigma = st.slider(
        "Threshold Multiplier (k)",
        1.5,
        6.0,
        float(preset["k_sigma"]),
        step=0.25,
        key=f"k_{selected_scenario}",
    )
    delta_opts = sorted({float(d) for d in CFG["dashboard"]["adwin_delta_options"]} | {float(preset["adwin_delta"])})
    adwin_delta = st.select_slider(
        "Drift Sensitivity (δ)",
        options=delta_opts,
        value=float(preset["adwin_delta"]),
        key=f"d_{selected_scenario}",
    )

    if selected_scenario == "Custom Dataset / Inject Drift":
        st.markdown("---")
        st.markdown("### Drift & Fault Injection")
        drift_type = st.selectbox("Drift Type", ["None", "Sudden Step", "Gradual Ramp", "Recurring Toggle"])
        drift_start = st.slider(
            "Drift Start (t)",
            warmup_pts,
            max(warmup_pts + 1, n_samples - 5),
            min(max(warmup_pts + 10, n_samples // 2), n_samples - 5),
            disabled=(drift_type == "None"),
        )
        drift_sigma = st.slider("Drift Magnitude (× σ)", -5.0, 5.0, 2.0, 0.25, disabled=(drift_type == "None"))
        inject_spike = st.checkbox("Inject Point Fault", value=False)
        spike_idx = st.slider("Fault Step (t)", warmup_pts, n_samples - 1, min(warmup_pts + 20, n_samples - 1), disabled=not inject_spike)
        spike_sigma = st.slider("Fault Size (× σ)", 2.0, 12.0, 6.0, 0.5, disabled=not inject_spike)
        stream_df = apply_injections(
            base_df, drift_type, drift_start, drift_sigma * value_std, inject_spike, spike_idx, spike_sigma * value_std
        )
    else:
        stream_df = base_df


# ---------------------------------------------------------------------------
# Run Detector
# ---------------------------------------------------------------------------
results_df, model_comparison = run_evaluation(
    stream_df,
    warmup=warmup_pts,
    k_sigma=k_sigma,
    adwin_delta=adwin_delta,
    drift_method=drift_method,
    cfg=CFG,
)

active_key = {
    "adwin": "HST + ADWIN (Adaptive)",
    "page_hinkley": "HST + Page-Hinkley (Adaptive)",
    "kswin": "HST + KSWIN (Adaptive)",
}.get(drift_method, "HST + ADWIN (Adaptive)")

adap_m = model_comparison[active_key]
stat_m = model_comparison["Static Isolation Forest"]
adap_fp = int(adap_m["False Alarms"])
stat_fp = int(stat_m["False Alarms"])
n_drifts = int(results_df["drift"].sum())
gt_windows = contiguous_spans(results_df["label"].to_numpy() == 1, results_df["step"].to_numpy())
caught_windows = sum(1 for (s0, s1) in gt_windows if bool(results_df.loc[s0:s1, "alarm"].any()))


# ---------------------------------------------------------------------------
# 3 Clean Top-Level Tabs (One Idea Per Screen — Zero Clutter)
# ---------------------------------------------------------------------------
tab_stream, tab_compare, tab_details = st.tabs(
    [
        "Stream Detection",
        "Adaptive vs. Static & Ablations",
        "Scores & Event Log",
    ]
)

with tab_stream:
    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Real Faults Caught",
        f"{caught_windows} / {len(gt_windows)}" if gt_windows else "0 Faults in Stream",
    )
    m2.metric(
        "False Alarms",
        f"{adap_fp}",
        delta=f"{adap_fp - stat_fp} vs Static ({stat_fp})" if stat_fp != adap_fp else None,
        delta_color="inverse",
    )
    m3.metric(
        "NAB Benchmark Score",
        f"{adap_m['NAB Score']:.1f} / 100",
        delta=f"+{adap_m['NAB Score'] - stat_m['NAB Score']:.1f} vs Static"
        if adap_m["NAB Score"] != stat_m["NAB Score"]
        else None,
    )
    m4.metric("Drift Adaptations", f"{n_drifts}")

    st.write("")
    chart_slot = st.empty()
    if play_clicked:
        frame_ends = np.unique(np.linspace(min(warmup_pts, len(results_df)), len(results_df), 24, dtype=int))
        for end_idx in frame_ends:
            chart_slot.plotly_chart(
                build_single_stream_chart(results_df.iloc[:end_idx], warmup_pts, unit),
                width="stretch",
            )
            time.sleep(0.09)
    else:
        chart_slot.plotly_chart(
            build_single_stream_chart(results_df, warmup_pts, unit),
            width="stretch",
        )

    st.markdown(
        f'<div class="summary-line"><b>{selected_scenario} ({active_key}):</b> {preset["desc"]} '
        f'Static Isolation Forest raised <b>{stat_fp}</b> false alarms (NAB: <b>{stat_m["NAB Score"]:.1f}</b>) vs. '
        f'<b>{adap_fp}</b> false alarms (NAB: <b>{adap_m["NAB Score"]:.1f}</b>, F1: <b>{adap_m["F1 Score"]:.3f}</b>) with adaptive streaming detection.</div>',
        unsafe_allow_html=True,
    )

with tab_compare:
    st.write("")
    st.plotly_chart(build_static_vs_adaptive_chart(results_df, warmup_pts, unit), width="stretch")

    st.write("")
    st.markdown("#### Multi-Detector & Drift-Adaptation Benchmark")
    comp_df = pd.DataFrame.from_dict(model_comparison, orient="index").reset_index().rename(columns={"index": "Detector"})
    st.dataframe(comp_df, width="stretch", hide_index=True)

    with st.expander("Threshold Sensitivity & Ablation Analysis (k_sigma Sweep)", expanded=False):
        sweep_df = run_sensitivity_sweep(
            stream_df, warmup=warmup_pts, adwin_delta=adwin_delta, drift_method=drift_method, cfg=CFG
        )
        st.plotly_chart(build_sensitivity_chart(sweep_df), width="stretch")
        st.dataframe(sweep_df, width="stretch", hide_index=True)

with tab_details:
    st.write("")
    hdr_c1, hdr_c2 = st.columns([3.0, 1.2])
    with hdr_c1:
        st.markdown("#### Anomaly Score vs. Adaptive Threshold")
    with hdr_c2:
        show_parts = st.checkbox("Show HST & Z-Score Components", value=False)
    st.plotly_chart(build_score_chart(results_df, warmup_pts, show_components=show_parts), width="stretch")

    st.write("")
    ev_c1, ev_c2 = st.columns([3.2, 1.0])
    with ev_c1:
        st.markdown("#### Notable Events (Alarms & Drift Shifts)")
    with ev_c2:
        st.download_button(
            "⬇ Export Telemetry CSV",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name=f"telemetry_results_{selected_scenario.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            width="stretch",
        )
    ev_df = results_df.loc[
        results_df["alarm"] | results_df["drift"],
        ["step", "timestamp", "value", "label", "score", "threshold", "alarm", "drift"],
    ].copy()
    if len(ev_df) == 0:
        st.info("No alarms or drift adaptations occurred in this stream.")
    else:
        ev_df["Event"] = np.where(
            ev_df["alarm"] & (ev_df["label"] == 1),
            "True Fault Caught",
            np.where(ev_df["drift"], "Drift Adapted", "False Alarm"),
        )
        st.dataframe(
            ev_df[["step", "timestamp", "Event", "value", "score", "threshold"]],
            width="stretch",
            hide_index=True,
            height=260,
        )
