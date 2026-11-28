# Drift-Adaptive Unsupervised Anomaly Detection in Streaming Sensor Data

**Course:** AIML Lab (`IS353IA`) — Experiential Learning (Phase 1 & 2)  
**Institution:** RV College of Engineering, Department of CSE  
**Team:** Mohith S D Gowda (`1RV24CS158`) | Navaneeth V (`1RV24CS165`)  
**Guide:** Dr. Veena Gadad  

---

## Project Overview

Industrial and defense sensor streams are unbounded, unlabelled, and subject to **concept drift** (changes in normal operating baseline due to load, wear, or seasonality). This repository implements a single-pass, fixed-memory **online anomaly detection pipeline**:

1. **Streaming Scorer:** **Half-Space Trees (HST)** scores each incoming reading in $O(1)$ time and constant memory (`score_one`), combined with a rolling z-deviation component, then updates its mass profiles (`learn_one`) under a strict **prequential (test-then-train)** protocol.
2. **Cold-Start / Warm-up (`N₀` samples):** The first `N₀` samples establish the initial baseline score distribution ($\mu_0, \sigma_0$) and set the initial threshold $\text{thr}_0 = \mu_0 + k \cdot \sigma_0$.
3. **Interchangeable Drift Monitors (`ADWIN`, `Page-Hinkley`, `KSWIN`):** Three distribution-free sequential drift detectors monitor the non-alarm stream and detect statistically significant regime shifts, re-anchoring the threshold and baseline without offline retraining.
4. **Static & Non-Adaptive Baselines:** Compares the drift-adaptive detectors against a non-adaptive streaming **Plain HST** ablation and static **Isolation Forest** and **One-Class SVM** models trained on the warm-up window.
5. **Standardized Evaluation:** Computes **Window-Adjusted Precision / Recall / F1**, the normalized **Numenta Anomaly Benchmark (NAB) Score** in $[0, 100]$, **False Alarms per 1,000 samples**, **Mean Detection Delay**, **Single-Sample Latency (ms)**, and **Serialized Model Memory (KB)**.

---

## Datasets

| File | Source | Rows | Labelled-anomaly rows |
|------|--------|------|-----------------------|
| `nab/ambient_temperature_system_failure.csv` | NAB `realKnownCause` (the stream shown on PPT slide 2) | 7,267 | 726 |
| `nab/machine_temperature_system_failure.csv` | NAB `realKnownCause` | 22,695 | 2,268 |
| `nab/cpu_utilization_asg_misconfiguration.csv` | NAB `realKnownCause` | 18,050 | 1,499 |
| `normal.csv`, `anomaly.csv` | Synthetic test fixtures | 60 each | 0 / 1 |

NAB data: Numenta Anomaly Benchmark, https://github.com/numenta/NAB (AGPL-3.0 licence — cite it in the report).
Labels come from NAB's `combined_windows.json`: every row inside an official anomaly window is labelled `1`.
To regenerate `data/nab/`: `python scripts/prepare_nab.py`.

---

## Repository Structure

```text
Anomaly-Detection-With-Concept-Drift-Handling/
├── README.md                  # Project documentation and quick-start guide
├── config.toml                # Centralized hyperparameter configuration
├── requirements.txt           # Python dependencies
├── data/
│   ├── normal.csv             # 60-sample synthetic clean stream (test fixture)
│   ├── anomaly.csv            # 60-sample synthetic stream with an injected spike at t=30
│   ├── nab_raw/               # Original NAB files as downloaded (+ combined_windows.json labels)
│   └── nab/                   # NAB streams converted to timestamp,value,label
├── scripts/
│   ├── prepare_nab.py         # Converts nab_raw/ -> nab/ using official NAB anomaly windows
│   └── run_benchmark.py       # Reproducible CLI benchmark across all 3 NAB streams & 6 models
├── src/
│   ├── __init__.py
│   ├── config.py              # TOML configuration loader & override helper
│   ├── detector.py            # HST + (ADWIN / Page-Hinkley / KSWIN), Static baselines, NAB scorer
│   └── dashboard.py           # Interactive Streamlit dashboard, ablation & sensitivity UI
└── tests/
    └── test_detector.py       # Automated verification suite (6 unit & benchmark tests)
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt
```
The detector includes a native Half-Space Trees and ADWIN implementation, so
River and a Rust compiler are not required.

### 2. Launch the Formal Streamlit UI
```powershell
streamlit run src/dashboard.py

### 3. Run the Automated Verification Suite
```powershell
pytest tests/test_detector.py -v
```

### Python Streaming API
```python
from src.detector import OnlineAnomalySystem

detector = OnlineAnomalySystem(warmup=50, k_sigma=3.0, adwin_delta=0.002)

for reading in sensor_stream:
    res = detector.step({"value": reading})
    if res["alarm"]:
        print(f"🚨 Fault! score={res['score']:.3f} > thr={res['threshold']:.3f}")
```

---

## 🗂️ Project Structure

```text
├── assets/                # Animated SVG banner & benchmark plots
├── config.toml            # Hot-reloadable hyperparameters (HST, ADWIN, thresholds)
├── data/nab/              # Processed Numenta Anomaly Benchmark streams
├── scripts/prepare_nab.py # Converts raw NAB files + official anomaly windows
├── src/detector.py        # NativeHalfSpaceTrees, NativeADWIN & baselines
├── src/dashboard.py       # Interactive Streamlit replay, comparison & drift injector
└── tests/                 # Verification suite (normal, spike, and step-drift tests)
```

<details>
<summary><b>📚 References &amp; Dataset Licence</b></summary>

- **HST:** Tan, Ting & Liu (2011), *Fast anomaly detection for streaming data*, IJCAI.
- **ADWIN:** Bifet & Gavaldà (2007), *Learning from time-changing data with adaptive windowing*, SIAM SDM.
- **NAB:** Ahmad et al. (2017), *Unsupervised real-time anomaly detection for streaming data*, Neurocomputing ([Numenta/NAB](https://github.com/numenta/NAB), AGPL-3.0).
- **Benchmark & Drift:** Cao et al. (2025), *AI Review*; Gama et al. (2014), *ACM Computing Surveys*.

</details>

---

<div align="center">

**AIML Lab (`IS353IA`) — Experiential Learning** • **RV College of Engineering®, Dept. of CSE**  
**Team:** Mohith S D Gowda (`1RV24CS158`) · Navaneeth V (`1RV24CS165`) &nbsp;|&nbsp; **Guide:** Dr. Veena Gadad

</div>
