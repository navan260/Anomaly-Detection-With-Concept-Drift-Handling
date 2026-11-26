# Drift-Adaptive Unsupervised Anomaly Detection in Streaming Sensor Data

**Course:** AIML Lab (`IS353IA`) — Experiential Learning (Phase 1 & 2)  
**Institution:** RV College of Engineering, Department of CSE  
**Team:** Mohith S D Gowda (`1RV24CS158`) | Navaneeth V (`1RV24CS165`)  
**Guide:** Dr. Veena Gadad  

---

## Project Overview

Industrial and defense sensor streams are unbounded, unlabelled, and subject to **concept drift** (changes in normal operating baseline due to load, wear, or seasonality). This repository implements a single-pass, fixed-memory **online anomaly detection pipeline**:

1. **Streaming Scorer:** **Half-Space Trees (HST)** scores each incoming reading in $O(1)$ time and constant memory (`score_one`), then updates its mass profiles (`learn_one`) under a strict **prequential (test-then-train)** protocol.
2. **Cold-Start / Warm-up (`N₀` samples):** The first `N₀` samples establish the initial baseline score distribution ($\mu_0, \sigma_0$) and set the initial threshold $\text{thr}_0 = \mu_0 + k \cdot \sigma_0$.
3. **Drift Monitor (`ADWIN`):** An Adaptive Windowing monitor watches the stream and detects significant shifts in mean without distribution assumptions, immediately re-anchoring the threshold and baseline without offline retraining.
4. **Static Baselines:** Compares the adaptive model against static **Isolation Forest** and **One-Class SVM** models trained on the warm-up window.

---

## Datasets

| File | Source | Rows | Labelled-anomaly rows |
|------|--------|------|-----------------------|
| `nab/ambient_temperature_system_failure.csv` | NAB `realKnownCause` (the stream shown on PPT slide 3) | 7,267 | 726 |
| `nab/machine_temperature_system_failure.csv` | NAB `realKnownCause` | 22,695 | 2,268 |
| `nab/cpu_utilization_asg_misconfiguration.csv` | NAB `realKnownCause` | 18,050 | 1,499 |
| `normal.csv`, `anomaly.csv` | Synthetic test fixtures | 60 each | 0 / 1 |

NAB data: Numenta Anomaly Benchmark, https://github.com/numenta/NAB (AGPL-3.0 licence — cite it in the report).
Labels come from NAB's `combined_windows.json`: every row inside an official anomaly window is labelled `1`.
To regenerate `data/nab/`: `python scripts/prepare_nab.py`.

---

## Repository Structure

```text
aimlel/
├── README.md                  # Project documentation and quick-start guide
├── requirements.txt           # Python dependencies
├── data/
│   ├── normal.csv             # 60-sample synthetic clean stream (test fixture, generated)
│   ├── anomaly.csv            # 60-sample synthetic stream with an injected spike at t=30 (test fixture)
│   ├── nab_raw/               # Original NAB files as downloaded (+ combined_windows.json labels)
│   └── nab/                   # NAB streams converted to timestamp,value,label
├── scripts/
│   └── prepare_nab.py         # Converts nab_raw/ -> nab/ using official NAB anomaly windows
├── src/
│   ├── __init__.py
│   ├── detector.py            # OnlineAnomalySystem (HST + ADWIN) & StaticBaselineDetector
│   └── dashboard.py           # Minimal, formal Streamlit dashboard & benchmark UI
└── tests/
    └── test_detector.py       # Automated verification suite for normal, anomaly, and drift streams
```

---

## Quick Start

### 1. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```
The detector includes a native Half-Space Trees and ADWIN implementation, so
River and a Rust compiler are not required.

### 2. Launch the Formal Streamlit UI
```powershell
streamlit run src/dashboard.py
```

### 3. Run the Automated Verification Suite
```powershell
pytest tests/test_detector.py -v
```
