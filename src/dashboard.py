"""
Minimal, Clean, Formal Dashboard for Drift-Adaptive Streaming Anomaly Detection.
Run with: streamlit run src/dashboard.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CONFIG_PATH, load_config
from src.detector import OnlineAnomalySystem, StaticBaselineDetector

# Re-read on every rerun, so edits to config.toml apply after pressing "R"
CFG = load_config()


# ---------------------------------------------------------------------------
# Page Configuration & Formal Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Drift-Adaptive Streaming Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Formal, clean academic styling */
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        h1, h2, h3 {
            font-family: "Inter", "Segoe UI", Roboto, Helvetica, sans-serif;
            color: #0F172A;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        .header-meta {
            font-size: 0.875rem;
            color: #475569;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 0.85rem;
            margin-bottom: 1.25rem;
        }
        .header-badge {
            display: inline-block;
            background-color: #F1F5F9;
            color: #1E293B;
            border: 1px solid #CBD5E1;
            border-radius: 4px;
            padding: 0.15rem 0.5rem;
            font-size: 0.78rem;
            font-weight: 500;
            margin-right: 0.5rem;
        }
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.4rem;
            color: #0F172A;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Stream Preparation & Synthetic Drift / Fault Injection
# ---------------------------------------------------------------------------
NAB_STREAMS = {
    "NAB: Ambient Temperature System Failure (7,267 pts)": "ambient_temperature_system_failure.csv",
    "NAB: Machine Temperature System Failure (22,695 pts)": "machine_temperature_system_failure.csv",
    "NAB: CPU Utilization ASG Misconfiguration (18,050 pts)": "cpu_utilization_asg_misconfiguration.csv",
}


def load_base_stream(choice: str, uploaded_file) -> pd.DataFrame:
    data_dir = ROOT_DIR / "data"
    if choice in NAB_STREAMS:
        # Real Numenta Anomaly Benchmark stream; label = 1 inside official NAB anomaly windows
        df = pd.read_csv(data_dir / "nab" / NAB_STREAMS[choice])
    elif choice == "Uploaded CSV" and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif choice == "Sample: Normal Stream (data/normal.csv)":
        df = pd.read_csv(data_dir / "normal.csv")
    elif choice == "Sample: Anomaly Stream (data/anomaly.csv)":
        df = pd.read_csv(data_dir / "anomaly.csv")
    else:
        # Synthetic extended NAB-like stream (160 readings) for stress testing
        rng = np.random.default_rng(42)
        n = 160
        t = np.arange(n)
        vals = 0.11 + 0.02 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 0.012, size=n)
        labels = np.zeros(n, dtype=int)
        # Inject two point anomalies
        vals[55] = 1.65
        labels[55] = 1
        vals[125] = 1.95
        labels[125] = 1
        df = pd.DataFrame({"timestamp": t, "value": np.round(vals, 4), "label": labels})

    # Clean up column types in case comments exist in CSV
    df = df.copy()
    if "value" in df.columns and df["value"].dtype == object:
        df["value"] = (
            df["value"].astype(str).str.split("#").str[0].str.strip().astype(float)
        )
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    if "label" not in df.columns:
        df["label"] = 0
    return df


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
    if drift_type == "Sudden Drift (Step Shift)":
        vals[d_start:] += drift_magnitude
    elif drift_type == "Gradual Drift (Linear Ramp)":
        ramp_len = max(1, n - d_start)
        vals[d_start:] += np.linspace(0.0, drift_magnitude, ramp_len)
    elif drift_type == "Recurring Drift (Regime Toggle)":
        mid = d_start + max(5, (n - d_start) // 2)
        vals[d_start:mid] += drift_magnitude

    if inject_spike and 0 <= spike_idx < n:
        vals[spike_idx] += spike_magnitude
        labels[spike_idx] = 1

    out["value"] = vals
    out["label"] = labels
    return out


# ---------------------------------------------------------------------------
# Prequential Evaluation Runner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running prequential evaluation over the stream...")
def run_prequential_stream(
    df: pd.DataFrame,
    warmup: int,
    k_sigma: float,
    adwin_delta: float,
    cfg: dict,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    values = df["value"].astype(float).to_numpy()
    timestamps = df["timestamp"].to_numpy()
    y_true = df["label"].astype(int).to_numpy()

    adaptive_sys = OnlineAnomalySystem(
        warmup=warmup, k_sigma=k_sigma, adwin_delta=adwin_delta, adaptive=True, cfg=cfg
    )
    static_hst = OnlineAnomalySystem(
        warmup=warmup, k_sigma=k_sigma, adwin_delta=adwin_delta, adaptive=False, cfg=cfg
    )
    iforest_base = StaticBaselineDetector(method="iforest", warmup=warmup, cfg=cfg)
    ocsvm_base = StaticBaselineDetector(method="ocsvm", warmup=warmup, cfg=cfg)

    # Proposed adaptive detector: true one-pass prequential replay
    records: List[dict] = []
    for idx, (val, ts, lbl) in enumerate(zip(values, timestamps, y_true)):
        res = adaptive_sys.step({"value": float(val)}, timestamp=ts)
        records.append(
            {
                "step": idx,
                "timestamp": ts,
                "value": float(val),
                "label": int(lbl),
                "score": round(res["score"], 4),
                "threshold": round(res["threshold"], 4),
                "baseline_mean": round(res["baseline_mean"], 4),
                "alarm": bool(res["alarm"]),
                "drift": bool(res["drift"]),
                "warmup": bool(res["warmup"]),
                "latency_ms": round(res["latency_ms"], 4),
            }
        )
    res_df = pd.DataFrame(records)

    # Ablation: same HST scorer with drift adaptation switched off
    t0 = time.perf_counter()
    static_hst_alarms = np.array(
        [int(static_hst.step({"value": float(v)})["alarm"]) for v in values]
    )
    static_hst_lat = (time.perf_counter() - t0) * 1000.0 / max(len(values), 1)

    # Static offline baselines (frozen after warm-up, so scored in one batch)
    iforest_alarms, iforest_lat = iforest_base.run_stream(values)
    ocsvm_alarms, ocsvm_lat = ocsvm_base.run_stream(values)

    def state_kb(model) -> float:
        """Size of the model's serialised state — what must be kept in memory."""
        try:
            return round(len(pickle.dumps(model)) / 1024.0, 1)
        except Exception:
            return float("nan")

    def summarize(preds: np.ndarray, latency_ms: float, model) -> Dict[str, float]:
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if np.sum(y_true) == 0 and fp == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(y_true) == 0 else 0.0)
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "Alarms Raised": int(np.sum(preds)),
            "True Positives": tp,
            "False Alarms": fp,
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1 Score": round(f1, 3),
            "Mean Latency (ms)": round(float(latency_ms), 3),
            "Model State (KB)": state_kb(model),
        }

    comparison = {
        "HST + ADWIN (Proposed Adaptive)": summarize(
            res_df["alarm"].astype(int).to_numpy(), res_df["latency_ms"].mean(), adaptive_sys
        ),
        "Half-Space Trees (No Drift Adapt)": summarize(static_hst_alarms, static_hst_lat, static_hst),
        "Static Isolation Forest (Offline Warm-up)": summarize(iforest_alarms, iforest_lat, iforest_base),
        "Static One-Class SVM (Offline Warm-up)": summarize(ocsvm_alarms, ocsvm_lat, ocsvm_base),
    }

    return res_df, comparison


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Stream Configuration")
    dataset_choice = st.selectbox(
        "Select Data Stream",
        [
            *NAB_STREAMS.keys(),
            "Sample: Anomaly Stream (data/anomaly.csv)",
            "Sample: Normal Stream (data/normal.csv)",
            "Synthetic: Extended Stream (160 pts)",
            "Uploaded CSV",
        ],
    )
    uploaded_csv = None
    if dataset_choice == "Uploaded CSV":
        uploaded_csv = st.file_uploader("Upload CSV (`value` column required)", type=["csv"])

    base_df = load_base_stream(dataset_choice, uploaded_csv)
    if len(base_df) > 1000:
        max_pts = st.slider(
            "Samples to Replay (from start of stream)",
            min_value=1000,
            max_value=len(base_df),
            value=min(len(base_df), int(CFG["dashboard"]["default_replay_samples"])),
            step=500,
            help="Long NAB streams can be truncated to keep the replay responsive.",
        )
        base_df = base_df.head(max_pts).reset_index(drop=True)
    n_samples = len(base_df)
    # Scale for drift/fault sliders so they make sense for any sensor unit (e.g. °F, % CPU)
    value_std = max(float(base_df["value"].std()), 0.05)

    st.markdown("---")
    st.markdown("### Model & Threshold Parameters")
    st.caption(f"Defaults loaded from `{CONFIG_PATH.name}`. Sliders override them for this session only.")
    w_cfg = CFG["warmup"]
    min_warmup = int(w_cfg["min_samples"])
    max_warmup = max(min_warmup + 5, n_samples // 2)
    default_warmup = int(w_cfg["samples_long_stream"]) if n_samples > 1000 else int(w_cfg["samples_short_stream"])
    default_warmup = min(max(default_warmup, min_warmup), max_warmup)
    warmup_pts = st.slider(
        "Warm-up Window (N₀ samples)",
        min_value=min_warmup,
        max_value=max_warmup,
        value=default_warmup,
        help="Initial readings used to establish baseline mean (μ₀) and threshold (μ₀ + k·σ₀).",
    )
    k_sigma = st.slider(
        "Threshold Multiplier (k · σ)",
        min_value=1.0,
        max_value=8.0,
        value=float(CFG["threshold"]["k_sigma"]),
        step=0.25,
        help="Controls sensitivity of the adaptive threshold.",
    )
    delta_options = sorted({float(d) for d in CFG["dashboard"]["adwin_delta_options"]} | {float(CFG["adwin"]["delta"])})
    adwin_delta = st.select_slider(
        "ADWIN Drift Confidence (δ)",
        options=delta_options,
        value=float(CFG["adwin"]["delta"]),
        help="Smaller δ makes ADWIN stricter; larger δ makes drift adaptation more sensitive.",
    )

    st.markdown("---")
    st.markdown("### Drift & Fault Injector")
    drift_type = st.selectbox(
        "Concept Drift Regime",
        [
            "None",
            "Sudden Drift (Step Shift)",
            "Gradual Drift (Linear Ramp)",
            "Recurring Drift (Regime Toggle)",
        ],
    )
    drift_start = st.slider(
        "Drift Onset Index (t)",
        min_value=warmup_pts,
        max_value=max(warmup_pts + 1, n_samples - 5),
        value=min(max(warmup_pts + 5, n_samples // 2), n_samples - 5),
        disabled=(drift_type == "None"),
    )
    drift_sigma = st.slider(
        "Drift Shift Magnitude (× stream σ)",
        min_value=-5.0,
        max_value=5.0,
        value=float(CFG["dashboard"]["drift_magnitude_sigma"]),
        step=0.25,
        disabled=(drift_type == "None"),
        help=f"Expressed in multiples of the stream's standard deviation (σ ≈ {value_std:.3g}).",
    )
    drift_mag = drift_sigma * value_std

    inject_spike = st.checkbox("Inject Additional Point Fault", value=False)
    spike_idx = st.slider(
        "Fault Timestamp Index",
        min_value=warmup_pts,
        max_value=n_samples - 1,
        value=min(warmup_pts + 15, n_samples - 1),
        disabled=not inject_spike,
    )
    spike_sigma = st.slider(
        "Fault Spike Magnitude (× stream σ)",
        min_value=1.0,
        max_value=15.0,
        value=float(CFG["dashboard"]["spike_magnitude_sigma"]),
        step=0.5,
        disabled=not inject_spike,
    )
    spike_mag = spike_sigma * value_std


# ---------------------------------------------------------------------------
# Main Header
# ---------------------------------------------------------------------------
st.markdown("## Drift-Adaptive Unsupervised Anomaly Detection in Streaming Sensor Data")
st.markdown(
    """
    <div class="header-meta">
        <span class="header-badge">AIML Lab (IS353IA)</span>
        <span class="header-badge">Prequential Test-Then-Train</span>
        <span class="header-badge">Half-Space Trees + ADWIN</span>
        RV College of Engineering &nbsp;|&nbsp; Mohith S D Gowda (1RV24CS158) &amp; Navaneeth V (1RV24CS165) &nbsp;|&nbsp; Guide: Dr. Veena Gadad
    </div>
    """,
    unsafe_allow_html=True,
)

# Run Pipeline
stream_df = apply_injections(
    base_df, drift_type, drift_start, drift_mag, inject_spike, spike_idx, spike_mag
)
results_df, model_comparison = run_prequential_stream(
    stream_df, warmup=warmup_pts, k_sigma=k_sigma, adwin_delta=adwin_delta, cfg=CFG
)

# ---------------------------------------------------------------------------
# Top Summary Metrics Row
# ---------------------------------------------------------------------------
adap_stats = model_comparison["HST + ADWIN (Proposed Adaptive)"]
n_alarms = int(results_df["alarm"].sum())
n_drifts = int(results_df["drift"].sum())
mean_lat = float(results_df["latency_ms"].mean())
final_thr = float(results_df["threshold"].iloc[-1])

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Samples Processed", f"{len(results_df)}")
m2.metric("Anomalies Flagged", f"{n_alarms}")
m3.metric("ADWIN Drift Shifts", f"{n_drifts}")
m4.metric("False Alarms", f"{adap_stats['False Alarms']}")
m5.metric("Active Threshold", f"{final_thr:.3f}")
m6.metric("Latency / Sample", f"{mean_lat:.3f} ms")


# ---------------------------------------------------------------------------
# Synchronized Dual-Panel Plotly Figure (Sensor Stream + Score & Threshold)
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    row_heights=[0.55, 0.45],
    subplot_titles=(
        "Sensor Reading Stream, Warm-up Window & Detected Anomalies",
        "Prequential Anomaly Score vs. Adaptive Threshold (μ + k·σ)",
    ),
)

# Shaded Warm-up region
warmup_end_t = results_df["step"].iloc[min(warmup_pts - 1, len(results_df) - 1)]
fig.add_vrect(
    x0=0,
    x1=warmup_end_t,
    fillcolor="#F1F5F9",
    opacity=0.8,
    layer="below",
    line_width=1,
    line_dash="dot",
    line_color="#94A3B8",
    annotation_text="Warm-up (N₀)",
    annotation_position="top left",
    row="all",
    col=1,
)

# Panel 1: Sensor value + tracked baseline + anomaly markers
fig.add_trace(
    go.Scatter(
        x=results_df["step"],
        y=results_df["value"],
        mode="lines",
        name="Sensor Value",
        line=dict(color="#1E293B", width=1.8),
    ),
    row=1,
    col=1,
)

post_warmup = results_df[~results_df["warmup"]]
fig.add_trace(
    go.Scatter(
        x=post_warmup["step"],
        y=post_warmup["baseline_mean"],
        mode="lines",
        name="Adaptive Baseline (μ)",
        line=dict(color="#64748B", width=1.3, dash="dot"),
    ),
    row=1,
    col=1,
)

alarms_df = results_df[results_df["alarm"]]
fig.add_trace(
    go.Scatter(
        x=alarms_df["step"],
        y=alarms_df["value"],
        mode="markers",
        name="Flagged Anomaly",
        marker=dict(color="#DC2626", size=9, symbol="circle", line=dict(color="#FFFFFF", width=1.5)),
    ),
    row=1,
    col=1,
)

# Panel 2: Anomaly Score + Adaptive Threshold
fig.add_trace(
    go.Scatter(
        x=results_df["step"],
        y=results_df["score"],
        mode="lines",
        name="Anomaly Score (sₜ)",
        line=dict(color="#2563EB", width=1.7),
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=post_warmup["step"],
        y=post_warmup["threshold"],
        mode="lines",
        name="Adaptive Threshold (thrₜ)",
        line=dict(color="#D97706", width=1.8, dash="dash"),
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=alarms_df["step"],
        y=alarms_df["score"],
        mode="markers",
        name="Score Exceeded Threshold",
        marker=dict(color="#DC2626", size=8, symbol="diamond"),
        showlegend=False,
    ),
    row=2,
    col=1,
)

# Group nearby drift detections into regions so the chart shows regime changes
# without drawing a full-height line for every individual ADWIN signal.
drifts_df = results_df[results_df["drift"]]
drift_steps = [int(step) for step in drifts_df["step"]]
drift_groups: List[List[int]] = []
for step in drift_steps:
    if not drift_groups or step - drift_groups[-1][-1] > 15:
        drift_groups.append([step])
    else:
        drift_groups[-1].append(step)

for group in drift_groups:
    start = group[0]
    end = group[-1]
    fig.add_vrect(
        x0=start,
        x1=end if end > start else start + 1,
        fillcolor="#0D9488",
        opacity=0.08,
        layer="below",
        line_width=0,
        row="all",
        col=1,
    )

if drift_groups:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="ADWIN drift region",
            marker=dict(color="#0D9488", size=9, symbol="square"),
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

fig.update_layout(
    height=650,
    margin=dict(l=20, r=20, t=70, b=35),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Inter, Segoe UI, sans-serif", size=12, color="#1E293B"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1.0,
        bgcolor="rgba(255,255,255,0.9)",
    ),
    hovermode="x",
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="#F1F5F9",
    linecolor="#CBD5E1",
    title_text="Stream Index (t)",
    rangeslider=dict(visible=True, thickness=0.06),
    row=2,
    col=1,
)
fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", title_text="Reading", row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", title_text="Score", row=2, col=1)

st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Bottom Tabs: Event Log | Baseline Benchmark | Prequential Architecture
# ---------------------------------------------------------------------------
tab_events, tab_bench, tab_arch = st.tabs(
    [
        "Alert & Drift Event Log",
        "Baseline Comparison (Static vs. Adaptive)",
        "Methodology & Cold-Start Protocol",
    ]
)

with tab_events:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Flagged Anomaly & Drift Events")
        events_view = results_df[results_df["alarm"] | results_df["drift"]][
            ["step", "timestamp", "value", "score", "threshold", "alarm", "drift", "latency_ms"]
        ]
        if len(events_view) == 0:
            st.info("No anomalies or drift shifts triggered in the current stream.")
        else:
            st.dataframe(events_view, width="stretch", hide_index=True)

    with col_b:
        st.markdown("#### Complete Stream Inspection")
        st.dataframe(
            results_df[
                ["step", "timestamp", "value", "score", "threshold", "warmup", "alarm", "drift", "latency_ms"]
            ],
            width="stretch",
            hide_index=True,
            height=260,
        )

with tab_bench:
    st.markdown("#### Quantitative Comparison Under Identical Prequential Protocol")
    st.caption(
        "Static models (Isolation Forest, One-Class SVM) are fitted on the initial warm-up window (N₀) "
        "only. Try injecting Sudden or Gradual Drift in the sidebar to observe how static baselines flood "
        "operators with false alarms while HST + ADWIN adapts."
    )
    comp_df = pd.DataFrame.from_dict(model_comparison, orient="index").reset_index()
    comp_df = comp_df.rename(columns={"index": "Detector Architecture"})
    st.dataframe(comp_df, width="stretch", hide_index=True)

with tab_arch:
    st.markdown(
        """
        #### Prequential (Test-Then-Train) Pipeline
        1. **Cold-Start / Warm-up (`t < N₀`)**: Each reading is scored immediately by the Half-Space Trees ensemble (`score_one`) and learned (`learn_one`), while scores are accumulated to estimate the initial baseline distribution $\\mu_0, \\sigma_0$.
        2. **Initial Threshold Assignment (`t = N₀`)**: The alarm threshold is initialized to $\\text{thr}_0 = \\mu_0 + k \\cdot \\sigma_0$.
        3. **Single-Pass Online Detection (`t > N₀`)**: Every incoming sample is scored *before* updating the model weights, ensuring zero data leakage.
        4. **ADWIN Drift Adaptation**: `ADWIN` monitors the stream in a variable-length window. When a statistically significant regime shift is detected, the older sub-window is dropped and the baseline threshold adapts immediately without offline retraining.
        """
    )
