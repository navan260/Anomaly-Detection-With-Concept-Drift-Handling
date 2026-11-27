""" Drift-Adaptive Online Anomaly Detection Engine. """
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

try:
    from .config import load_config
except ImportError:  # running detector.py directly
    from config import load_config

# Optional River integration if installed in the environment
try:
    from river import anomaly as river_anomaly
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class NativeHalfSpaceTrees:
    """
    Exact implementation of Streaming Half-Space Trees (Tan, Ting & Liu, 2011)
    matching River's HalfSpaceTrees API (`score_one`, `learn_one`).
    Uses compact flat arrays for deterministic O(1) time and low fixed memory.
    """

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 100,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: int = 42,
    ):
        self.n_trees = n_trees
        self.height = height
        self.window_size = window_size
        self.limits = limits or {}
        self.rng = np.random.default_rng(seed)
        self._samples_seen = 0
        self._max_mass = window_size * (1 << height)
        self.features: List[str] = []
        self.feat_idx: Optional[np.ndarray] = None
        self.splits: Optional[np.ndarray] = None
        self.r_leaf: Optional[np.ndarray] = None
        self.l_leaf: Optional[np.ndarray] = None
        self._tree_rows = np.arange(n_trees, dtype=np.int32)

    def _build_recursive(
        self,
        tree_idx: int,
        node_idx: int,
        n_feat: int,
        work_ranges: List[Tuple[float, float]],
        depth: int,
    ) -> None:
        if depth >= self.height:
            return
        f_i = int(self.rng.integers(0, n_feat))
        low, high = work_ranges[f_i]
        mid = 0.5 * (low + high)
        self.feat_idx[tree_idx, node_idx] = f_i
        self.splits[tree_idx, node_idx] = mid

        left_ranges = list(work_ranges)
        left_ranges[f_i] = (low, mid)
        right_ranges = list(work_ranges)
        right_ranges[f_i] = (mid, high)

        self._build_recursive(tree_idx, 2 * node_idx + 1, n_feat, left_ranges, depth + 1)
        self._build_recursive(tree_idx, 2 * node_idx + 2, n_feat, right_ranges, depth + 1)

    def _init_trees(self, x: Dict[str, float]) -> None:
        self.features = list(x.keys())
        n_feat = len(self.features)
        work_ranges: List[Tuple[float, float]] = []
        for f in self.features:
            val = x[f]
            if f in self.limits:
                work_ranges.append(self.limits[f])
            else:
                span = max(abs(val) * 2.0, 2.0)
                q = float(self.rng.uniform(0.15, 0.85))
                low = val - 2.0 * q * span
                work_ranges.append((low, low + 2.0 * span))

        n_internal = (1 << self.height) - 1
        n_leaves = 1 << self.height
        self.feat_idx = np.zeros((self.n_trees, n_internal), dtype=np.int8)
        self.splits = np.zeros((self.n_trees, n_internal), dtype=np.float32)
        self.r_leaf = np.zeros((self.n_trees, n_leaves), dtype=np.int32)
        self.l_leaf = np.zeros((self.n_trees, n_leaves), dtype=np.int32)

        for t in range(self.n_trees):
            self._build_recursive(t, 0, n_feat, list(work_ranges), 0)

    def _leaf_indices(self, x: Dict[str, float]) -> np.ndarray:
        vals_arr = np.asarray([float(x.get(f, 0.0)) for f in self.features], dtype=np.float32)
        n_internal = (1 << self.height) - 1
        node = np.zeros(self.n_trees, dtype=np.int32)
        for _ in range(self.height):
            f_idx = self.feat_idx[self._tree_rows, node]
            sp = self.splits[self._tree_rows, node]
            node = 2 * node + 1 + (vals_arr[f_idx] >= sp)
        return node - n_internal

    def score_one(self, x: Dict[str, float]) -> float:
        """Return an anomaly score in [0, 1] where higher = more anomalous."""
        if self.feat_idx is None:
            return 0.5
        leaves = self._leaf_indices(x)
        masses = (
            self.r_leaf[self._tree_rows, leaves]
            if self._samples_seen >= self.window_size
            else self.l_leaf[self._tree_rows, leaves]
        )
        total_mass_score = float(masses.sum()) * (1 << self.height)
        avg_score = total_mass_score / (self.n_trees * max(self._max_mass, 1))
        return float(np.clip(1.0 - avg_score, 0.0, 1.0))

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update latest window mass and pivot windows every `window_size` readings."""
        if self.feat_idx is None:
            self._init_trees(x)
        leaves = self._leaf_indices(x)
        self.l_leaf[self._tree_rows, leaves] += 1
        self._samples_seen += 1
        if self._samples_seen % self.window_size == 0:
            self.r_leaf[:] = self.l_leaf
            self.l_leaf.fill(0)


class NativeADWIN:
    """
    Adaptive Windowing (Bifet & Gavalda, 2007) drift detector.
    Maintains a variable-length window of recent values and shrinks it when a
    statistically significant difference in means is observed between two sub-windows.
    """

    def __init__(self, delta: float = 0.002, min_window: int = 10, max_window: int = 400):
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window
        self.window: deque[float] = deque(maxlen=max_window)
        self.change_detected: bool = False
        self.n_detections: int = 0

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        self.change_detected = False
        n = len(self.window)
        if n < self.min_window:
            return False

        arr = np.array(self.window, dtype=float)
        min_sub = max(3, self.min_window // 3)
        delta_prime = max(self.delta / math.log(max(n, 2)), 1e-9)

        csum = np.cumsum(arr)
        csq = np.cumsum(arr * arr)
        n0 = np.arange(n - min_sub, min_sub - 1, -1)
        n1 = n - n0
        s0, q0 = csum[n0 - 1], csq[n0 - 1]
        s1, q1 = csum[-1] - s0, csq[-1] - q0
        mu0, mu1 = s0 / n0, s1 / n1
        var0 = np.maximum(q0 / n0 - mu0 ** 2, 0.0)
        var1 = np.maximum(q1 / n1 - mu1 ** 2, 0.0)
        pooled_var = np.maximum((n0 * var0 + n1 * var1) / n, 1e-4)
        m_inv = 1.0 / n0 + 1.0 / n1
        epsilon = np.sqrt(2.0 * m_inv * pooled_var * math.log(2.0 / delta_prime))
        hits = np.nonzero(np.abs(mu0 - mu1) > epsilon)[0]

        if hits.size:
            cut = int(n0[hits[0]])
            self.change_detected = True
            self.n_detections += 1
            for _ in range(cut):
                self.window.popleft()

        return self.change_detected


@dataclass
class StepResult:
    index: int
    timestamp: str | float
    value: float
    score: float
    threshold: float
    alarm: bool
    drift: bool
    warmup_active: bool
    latency_ms: float


class OnlineAnomalySystem:
    """
    Drift-Adaptive Unsupervised Streaming Anomaly Detector.

    Implements prequential evaluation (test-then-train):
    1. Score incoming reading using Half-Space Trees + rolling feature statistics.
    2. Compare score against adaptive threshold (established after warm-up).
    3. Feed non-transient stream dynamics to ADWIN; when sustained drift is
       signaled, adapt the baseline and threshold without offline retraining.
    4. Update the streaming model (`learn_one`).
    """

    def __init__(
        self,
        warmup: Optional[int] = None,
        k_sigma: Optional[float] = None,
        adwin_delta: Optional[float] = None,
        adaptive: Optional[bool] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        cfg = cfg if cfg is not None else load_config()
        w, h, s, t, a = cfg["warmup"], cfg["hst"], cfg["score"], cfg["threshold"], cfg["adwin"]

        self.warmup = max(int(w["min_samples"]), int(warmup if warmup is not None else w["samples_short_stream"]))
        self.k = float(k_sigma if k_sigma is not None else t["k_sigma"])
        self.adwin_delta = float(adwin_delta if adwin_delta is not None else a["delta"])
        self.adaptive = bool(adaptive if adaptive is not None else a["adaptive"])
        self.window_size = int(h["window_size"])

        self.hst_weight = float(s["hst_weight"])
        self.z_decay = float(s["z_decay"])
        self.min_values_for_z = int(s["min_values_for_z"])
        self.thr_min = float(t["min_threshold"])
        self.thr_max = float(t["max_threshold"])
        self.init_std_floor = float(t["init_score_std_floor"])
        self.score_std_floor = float(t["score_std_floor"])
        self.ema_alpha = float(t["baseline_ema_alpha"])
        self.baseline_std_floor = float(t["baseline_std_floor"])
        self.post_drift_min_scores = int(t["post_drift_min_scores"])

        hst_kwargs = dict(
            n_trees=int(h["n_trees"]),
            height=int(h["height"]),
            window_size=self.window_size,
            seed=int(h["seed"]),
        )
        if HAS_RIVER and hasattr(river_anomaly, "HalfSpaceTrees"):
            self.hst = river_anomaly.HalfSpaceTrees(**hst_kwargs)
        else:
            self.hst = NativeHalfSpaceTrees(**hst_kwargs)

        max_w = max(int(a["max_window"]), self.warmup)
        min_w = max(int(a["min_window"]), min(self.warmup, 120) if self.warmup >= 100 else self.warmup // 2)
        self.adwin = NativeADWIN(
            delta=self.adwin_delta,
            min_window=min_w,
            max_window=max_w,
        )

        self.init_scores: List[float] = []
        self.init_values: List[float] = []
        self.score_window: deque[float] = deque(maxlen=max(int(t["score_window_min"]), self.window_size * 2))
        self.value_window: deque[float] = deque(maxlen=max(int(t["value_window_min"]), self.window_size))
        self.regime_window: deque[float] = deque(maxlen=max_w)

        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self.threshold: Optional[float] = None
        self._warmup_threshold: Optional[float] = None
        self.step_count: int = 0
        self._alarm_streak: List[float] = []
        self._streak_limit: int = 5

    def _compute_composite_score(self, val: float, hst_raw: float) -> float:
        if len(self.value_window) >= self.min_values_for_z:
            z_dev = abs(val - self.baseline_mean) / max(self.baseline_std, 1e-4)
            z_norm = 1.0 - math.exp(-self.z_decay * z_dev)
            return float(self.hst_weight * hst_raw + (1.0 - self.hst_weight) * z_norm)
        return float(hst_raw)

    def _effective_std(self) -> float:
        local_std = float(np.std(self.value_window)) if len(self.value_window) >= 2 else self.baseline_std
        if len(self.regime_window) >= 10:
            regime_std = float(np.std(self.regime_window))
            return max(local_std, 0.55 * regime_std, self.baseline_std_floor)
        return max(local_std, self.baseline_std_floor)

    def _update_threshold(self, scores: List[float]) -> None:
        mu_s = float(np.mean(scores))
        sd_s = max(float(np.std(scores)), self.score_std_floor)
        self.threshold = float(np.clip(mu_s + self.k * sd_s, self.thr_min, self.thr_max))

    def _set_initial_threshold(self) -> None:
        val_arr = np.array(self.init_values, dtype=float)
        if len(val_arr) > 0:
            self.baseline_mean = float(val_arr.mean())
            self.baseline_std = max(float(val_arr.std()), self.baseline_std_floor)
            for v in val_arr:
                self.adwin.window.append(float(v))
                self.regime_window.append(float(v))

        # Calibrate initial threshold on the warm-up readings using the warm-up-fitted HST & baseline
        # (same protocol as StaticBaselineDetector, avoiding empty-tree cold-start score inflation)
        calibrated_scores: List[float] = []
        prev = float(val_arr[0]) if len(val_arr) else 0.0
        for v in val_arr:
            fv = float(v)
            hs = float(self.hst.score_one({"value": fv, "roc": fv - prev}))
            calibrated_scores.append(self._compute_composite_score(fv, hs))
            prev = fv

        arr = np.array(calibrated_scores, dtype=float)
        mu = float(arr.mean())
        sigma = max(float(arr.std()), self.init_std_floor)
        self.threshold = float(np.clip(mu + self.k * sigma, self.thr_min, self.thr_max))
        self._warmup_threshold = self.threshold
        self.score_window.clear()
        self.score_window.extend(calibrated_scores[-self.score_window.maxlen :])

    def step(self, x: Dict[str, float], timestamp: str | float | None = None) -> dict:
        t0 = time.perf_counter()
        val = float(x.get("value", next(iter(x.values()))))

        if self.threshold is None and len(self.value_window) > 0:
            val_arr = np.array(self.value_window, dtype=float)
            self.baseline_mean = float(val_arr.mean())
            self.baseline_std = max(float(val_arr.std()), self.baseline_std_floor)

        prev_val = self.value_window[-1] if self.value_window else val
        features = {"value": val, "roc": val - prev_val}

        # 1. Test (Score) first - Prequential evaluation
        hst_score = float(self.hst.score_one(features))
        score = self._compute_composite_score(val, hst_score)

        # 2. Warm-up handling
        if self.threshold is None:
            self.value_window.append(val)
            self.init_values.append(val)
            self.init_scores.append(score)
            self.score_window.append(score)
            self.hst.learn_one(features)
            self.step_count += 1
            if len(self.init_scores) >= self.warmup:
                self._set_initial_threshold()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "score": score,
                "threshold": self.threshold if self.threshold is not None else 0.0,
                "alarm": False,
                "drift": False,
                "warmup": True,
                "latency_ms": latency_ms,
                "baseline_mean": self.baseline_mean,
            }

        # 3. Prequential alarm decision against active threshold (never retroactively erased)
        alarm = bool(score > self.threshold)
        drift = False

        # Winsorized long-window variance tracker prevents brief quiet spells from collapsing std
        if self.adaptive:
            clipped = float(
                np.clip(
                    val,
                    self.baseline_mean - 4.5 * self.baseline_std,
                    self.baseline_mean + 4.5 * self.baseline_std,
                )
            )
            self.regime_window.append(clipped)

        # 4. Gate ADWIN on isolated spikes; flush if a shift persists >= _streak_limit steps
        if not alarm:
            self._alarm_streak.clear()
            drift = self.adwin.update(val)
        else:
            self._alarm_streak.append(val)
            if len(self._alarm_streak) >= self._streak_limit:
                for sv in self._alarm_streak:
                    if self.adwin.update(sv):
                        drift = True
                self._alarm_streak.clear()

        if drift and self.adaptive:
            recent_vals = list(self.adwin.window)
            if len(recent_vals) >= 3:
                self.baseline_mean = float(np.mean(recent_vals))
                self.value_window.clear()
                self.value_window.extend(recent_vals)
                self.baseline_std = self._effective_std()
            adapted_score = self._compute_composite_score(val, hst_score)
            self.score_window.append(adapted_score)
            recent_scores = list(self.score_window)[-max(self.post_drift_min_scores, len(recent_vals)):]
            if recent_scores:
                self._update_threshold(recent_scores)

        # 5. Train (Learn) - update baseline statistics from ADWIN's active regime window
        if not alarm:
            self.value_window.append(val)
            self.score_window.append(score)
            if self.adaptive and len(self.adwin.window) >= self.min_values_for_z:
                self.baseline_mean = float(np.mean(self.adwin.window))
                self.baseline_std = self._effective_std()
                self._update_threshold(list(self.score_window))

        self.hst.learn_one(features)
        self.step_count += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "score": score,
            "threshold": float(self.threshold),
            "alarm": alarm,
            "drift": drift,
            "warmup": False,
            "latency_ms": latency_ms,
            "baseline_mean": self.baseline_mean,
        }


class StaticBaselineDetector:
    """
    Offline static baseline detector (Isolation Forest or One-Class SVM)
    trained only on the initial warm-up window, demonstrating how static models
    suffer from false-alarm floods under concept drift (PPT Slide 3 & 6).
    """

    def __init__(
        self,
        method: str = "iforest",
        warmup: Optional[int] = None,
        contamination: Optional[float] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        cfg = cfg if cfg is not None else load_config()
        b, w = cfg["baselines"], cfg["warmup"]
        self.method = method
        self.warmup = max(int(w["min_samples"]), int(warmup if warmup is not None else w["samples_short_stream"]))
        self.contamination = float(contamination if contamination is not None else b["contamination"])
        self.k = float(b["k_sigma"])
        self.z_weight = float(b["z_weight"])
        self.z_decay = float(cfg["score"]["z_decay"])
        self.init_X: List[List[float]] = []
        self.fitted = False
        self.mu0 = 0.0
        self.sigma0 = 1.0
        self.threshold = 0.0
        if method == "ocsvm":
            self.model = OneClassSVM(
                kernel=b["ocsvm_kernel"], nu=self.contamination, gamma=b["ocsvm_gamma"]
            )
        else:
            self.model = IsolationForest(
                n_estimators=int(b["iforest_n_estimators"]),
                contamination=self.contamination,
                random_state=int(b["random_state"]),
            )

    def _raw_score(self, X_arr: np.ndarray) -> np.ndarray:
        model_s = -self.model.decision_function(X_arr)
        z_s = np.abs(X_arr[:, 0] - self.mu0) / max(self.sigma0, 1e-3)
        return (1.0 - self.z_weight) * model_s + self.z_weight * (1.0 - np.exp(-self.z_decay * z_s))

    def step(self, x: Dict[str, float]) -> dict:
        t0 = time.perf_counter()
        val = float(x.get("value", next(iter(x.values()))))

        if not self.fitted:
            self.init_X.append([val])
            if len(self.init_X) >= self.warmup:
                X_arr = np.array(self.init_X, dtype=float)
                self.mu0 = float(X_arr.mean())
                self.sigma0 = max(float(X_arr.std()), 0.015)
                self.model.fit(X_arr)
                raw_scores = self._raw_score(X_arr)
                self.threshold = float(np.mean(raw_scores) + self.k * max(np.std(raw_scores), 0.02))
                self.fitted = True
            return {
                "score": 0.0,
                "threshold": self.threshold,
                "alarm": False,
                "drift": False,
                "warmup": True,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }

        raw_score = float(self._raw_score(np.array([[val]], dtype=float))[0])
        alarm = bool(raw_score > self.threshold)
        return {
            "score": raw_score,
            "threshold": self.threshold,
            "alarm": alarm,
            "drift": False,
            "warmup": False,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    def _measure_single_step_latency_ms(self, post_warmup_values: np.ndarray) -> float:
        """Measure true one-sample-at-a-time streaming inference latency (`step()`)."""
        if not self.fitted or len(post_warmup_values) == 0:
            return 0.0
        probe = post_warmup_values[: min(80, len(post_warmup_values))]
        t0 = time.perf_counter()
        for v in probe:
            self.step({"value": float(v)})
        return (time.perf_counter() - t0) * 1000.0 / len(probe)

    def run_stream(self, values: np.ndarray) -> Tuple[np.ndarray, float]:
        values = np.asarray(values, dtype=float)
        alarms = np.zeros(len(values), dtype=int)
        for v in values[: self.warmup]:
            self.step({"value": float(v)})
        rest = values[self.warmup:]
        if len(rest) and self.fitted:
            scores = self._raw_score(rest.reshape(-1, 1))
            alarms[self.warmup:] = (scores > self.threshold).astype(int)
        mean_latency = self._measure_single_step_latency_ms(rest)
        return alarms, mean_latency

    def run_stream_detailed(self, values: np.ndarray) -> Dict[str, Any]:
        """Return full per-step scores, alarms, frozen baseline, and true streaming latency."""
        values = np.asarray(values, dtype=float)
        n = len(values)
        alarms = np.zeros(n, dtype=bool)
        scores = np.zeros(n, dtype=float)
        w_end = min(self.warmup, n)
        for i in range(w_end):
            self.step({"value": float(values[i])})
        if self.fitted and w_end > 0:
            scores[:w_end] = self._raw_score(values[:w_end].reshape(-1, 1))
        rest = values[w_end:]
        if len(rest) and self.fitted:
            rest_scores = self._raw_score(rest.reshape(-1, 1))
            scores[w_end:] = rest_scores
            alarms[w_end:] = rest_scores > self.threshold
        mean_latency = self._measure_single_step_latency_ms(rest)
        return {
            "alarms": alarms,
            "scores": scores,
            "threshold": float(self.threshold),
            "baseline_mean": float(self.mu0),
            "mean_latency_ms": mean_latency,
        }
