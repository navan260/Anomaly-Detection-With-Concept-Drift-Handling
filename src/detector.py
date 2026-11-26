""" Drift-Adaptive Online Anomaly Detection Engine. """
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
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
    from river import drift as river_drift
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False


class _FallbackHalfSpaceNode:
    """Single node in a Half-Space Tree."""
    __slots__ = ("feature", "split_val", "left", "right", "r", "l")

    def __init__(self, feature: Optional[str] = None, split_val: float = 0.0,
                 left: Optional[_FallbackHalfSpaceNode] = None,
                 right: Optional[_FallbackHalfSpaceNode] = None):
        self.feature = feature
        self.split_val = split_val
        self.left = left
        self.right = right
        self.r = 0  # Reference window mass
        self.l = 0  # Latest window mass


class NativeHalfSpaceTrees:
    """
    Exact implementation of Streaming Half-Space Trees (Tan, Ting & Liu, 2011)
     matching River's HalfSpaceTrees API (`score_one`, `learn_one`).
    Provides deterministic, constant-time O(1) and constant-memory scoring.
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
        self.trees: List[_FallbackHalfSpaceNode] = []
        self._samples_seen = 0
        self._max_mass = window_size * (2 ** height)

    def _build_tree(
        self, features: List[str], work_ranges: Dict[str, Tuple[float, float]], depth: int
    ) -> _FallbackHalfSpaceNode:
        if depth >= self.height:
            return _FallbackHalfSpaceNode()
        feat = features[int(self.rng.integers(0, len(features)))]
        low, high = work_ranges[feat]
        split = (low + high) / 2.0

        left_ranges = dict(work_ranges)
        left_ranges[feat] = (low, split)
        right_ranges = dict(work_ranges)
        right_ranges[feat] = (split, high)

        return _FallbackHalfSpaceNode(
            feature=feat,
            split_val=split,
            left=self._build_tree(features, left_ranges, depth + 1),
            right=self._build_tree(features, right_ranges, depth + 1),
        )

    def _init_trees(self, x: Dict[str, float]) -> None:
        features = list(x.keys())
        work_ranges: Dict[str, Tuple[float, float]] = {}
        for f, val in x.items():
            if f in self.limits:
                work_ranges[f] = self.limits[f]
            else:
                span = max(abs(val) * 2.0, 2.0)
                q = float(self.rng.uniform(0.15, 0.85))
                low = val - 2.0 * q * span
                high = low + 2.0 * span
                work_ranges[f] = (low, high)
        self.trees = [
            self._build_tree(features, dict(work_ranges), 0) for _ in range(self.n_trees)
        ]

    def score_one(self, x: Dict[str, float]) -> float:
        """Return an anomaly score in [0, 1] where higher = more anomalous."""
        if not self.trees:
            return 0.5
        total_mass_score = 0.0
        for root in self.trees:
            node = root
            depth = 0
            while node.left is not None and node.right is not None:
                val = x.get(node.feature, 0.0)
                node = node.left if val < node.split_val else node.right
                depth += 1
            mass = node.r if self._samples_seen >= self.window_size else node.l
            total_mass_score += mass * (2 ** depth)

        avg_score = total_mass_score / (self.n_trees * max(self._max_mass, 1))
        # Anomaly score is 1 - normalized mass (sparse regions have low mass -> high anomaly score)
        return float(np.clip(1.0 - avg_score, 0.0, 1.0))

    def learn_one(self, x: Dict[str, float]) -> None:
        """Update latest window mass and pivot windows every `window_size` readings."""
        if not self.trees:
            self._init_trees(x)

        for root in self.trees:
            node = root
            node.l += 1
            while node.left is not None and node.right is not None:
                val = x.get(node.feature, 0.0)
                node = node.left if val < node.split_val else node.right
                node.l += 1

        self._samples_seen += 1
        if self._samples_seen % self.window_size == 0:
            for root in self.trees:
                self._pivot_node(root)

    def _pivot_node(self, node: Optional[_FallbackHalfSpaceNode]) -> None:
        if node is None:
            return
        node.r = node.l
        node.l = 0
        self._pivot_node(node.left)
        self._pivot_node(node.right)


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

        # Vectorised evaluation of every split point (older w0 | newer w1) using
        # cumulative sums, so each update costs one O(n) numpy pass.
        csum = np.cumsum(arr)
        csq = np.cumsum(arr * arr)
        n0 = np.arange(n - min_sub, min_sub - 1, -1)  # same order as scanning n1 upward
        n1 = n - n0
        s0, q0 = csum[n0 - 1], csq[n0 - 1]
        s1, q1 = csum[-1] - s0, csq[-1] - q0
        mu0, mu1 = s0 / n0, s1 / n1
        var0 = np.maximum(q0 / n0 - mu0 ** 2, 0.0)
        var1 = np.maximum(q1 / n1 - mu1 ** 2, 0.0)
        # Pooled within-subwindow variance (prevents the step jump itself from inflating variance)
        pooled_var = np.maximum((n0 * var0 + n1 * var1) / n, 1e-4)
        m_inv = 1.0 / n0 + 1.0 / n1
        epsilon = np.sqrt(2.0 * m_inv * pooled_var * math.log(2.0 / delta_prime))
        hits = np.nonzero(np.abs(mu0 - mu1) > epsilon)[0]

        if hits.size:
            cut = int(n0[hits[0]])
            self.change_detected = True
            self.n_detections += 1
            # Drop older sub-window w0 so window now holds only the new regime w1
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
    3. Feed baseline dynamics to ADWIN; if drift is signaled, adapt the baseline
       and threshold without offline retraining.
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
        """
        All hyperparameters come from `config.toml` (or the `cfg` dict passed in).
        The explicit keyword arguments, when given, override the config values.
        """
        cfg = cfg if cfg is not None else load_config()
        w, h, s, t, a = cfg["warmup"], cfg["hst"], cfg["score"], cfg["threshold"], cfg["adwin"]

        self.warmup = max(int(w["min_samples"]), int(warmup if warmup is not None else w["samples_short_stream"]))
        self.k = float(k_sigma if k_sigma is not None else t["k_sigma"])
        self.adwin_delta = float(adwin_delta if adwin_delta is not None else a["delta"])
        self.adaptive = bool(adaptive if adaptive is not None else a["adaptive"])
        self.window_size = int(h["window_size"])

        # Score / threshold constants
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

        # Initialize HST scorer (River if installed, otherwise NativeHST)
        hst_kwargs = dict(
            n_trees=int(h["n_trees"]), height=int(h["height"]),
            window_size=self.window_size, seed=int(h["seed"]),
        )
        if HAS_RIVER and hasattr(river_anomaly, "HalfSpaceTrees"):
            self.hst = river_anomaly.HalfSpaceTrees(**hst_kwargs)
        else:
            self.hst = NativeHalfSpaceTrees(**hst_kwargs)

        self.adwin = NativeADWIN(
            delta=self.adwin_delta,
            min_window=max(int(a["min_window"]), self.warmup // 2),
            max_window=int(a["max_window"]),
        )

        self.init_scores: List[float] = []
        self.score_window: deque[float] = deque(maxlen=max(int(t["score_window_min"]), self.window_size * 2))
        self.value_window: deque[float] = deque(maxlen=max(int(t["value_window_min"]), self.window_size))

        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self.threshold: Optional[float] = None
        self.step_count: int = 0

    def _compute_composite_score(self, val: float, hst_raw: float) -> float:
        """
        Combine HST subspace mass score with local rolling z-deviation so point
        anomalies and regime changes produce interpretable, well-scaled scores.
        """
        if len(self.value_window) >= self.min_values_for_z:
            z_dev = abs(val - self.baseline_mean) / max(self.baseline_std, 1e-4)
            z_norm = 1.0 - math.exp(-self.z_decay * z_dev)
            return float(self.hst_weight * hst_raw + (1.0 - self.hst_weight) * z_norm)
        return float(hst_raw)

    def _update_threshold(self, scores: List[float]) -> None:
        mu_s = float(np.mean(scores))
        sd_s = max(float(np.std(scores)), self.score_std_floor)
        self.threshold = float(np.clip(mu_s + self.k * sd_s, self.thr_min, self.thr_max))

    def _set_initial_threshold(self) -> None:
        arr = np.array(self.init_scores, dtype=float)
        mu = float(arr.mean())
        sigma = max(float(arr.std()), self.init_std_floor)
        self.threshold = float(mu + self.k * sigma)

        val_arr = np.array(self.value_window, dtype=float)
        if len(val_arr) > 0:
            self.baseline_mean = float(val_arr.mean())
            self.baseline_std = max(float(val_arr.std()), self.baseline_std_floor)

    def step(self, x: Dict[str, float], timestamp: str | float | None = None) -> dict:
        """
        Process a single sensor observation `x` (e.g. `{'value': 0.12}`) in one pass.
        Returns a dictionary containing `score`, `threshold`, `alarm`, `drift`,
        `warmup`, and `latency_ms`.
        """
        t0 = time.perf_counter()
        val = float(x.get("value", next(iter(x.values()))))

        # Rate of change feature for context
        prev_val = self.value_window[-1] if self.value_window else val
        features = {"value": val, "roc": val - prev_val}

        # 1. Test (Score) first - Prequential evaluation
        hst_score = float(self.hst.score_one(features))
        score = self._compute_composite_score(val, hst_score)

        # 2. Warm-up handling
        if self.threshold is None:
            self.value_window.append(val)
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

        # 3. Alarm comparison against active threshold
        alarm = bool(score > self.threshold)

        # 4. Drift monitoring (ADWIN watches stream statistics)
        drift = self.adwin.update(val)

        if drift and self.adaptive:
            # Re-anchor baseline and threshold to the post-drift window inside ADWIN
            recent_vals = list(self.adwin.window)
            if len(recent_vals) >= 3:
                self.baseline_mean = float(np.mean(recent_vals))
                self.baseline_std = max(float(np.std(recent_vals)), self.baseline_std_floor)
                self.value_window.clear()
                self.value_window.extend(recent_vals)
            # Re-evaluate score against the newly adapted regime
            score = self._compute_composite_score(val, hst_score)
            self.score_window.append(score)
            recent_scores = list(self.score_window)[-max(self.post_drift_min_scores, len(recent_vals)):]
            if recent_scores:
                self._update_threshold(recent_scores)
            alarm = bool(score > self.threshold)

        # 5. Train (Learn) - update baseline statistics
        self.value_window.append(val)
        if not alarm:
            self.score_window.append(score)
            if self.adaptive and len(self.value_window) >= self.min_values_for_z:
                self.baseline_mean = (1.0 - self.ema_alpha) * self.baseline_mean + self.ema_alpha * val
                self.baseline_std = max(float(np.std(self.value_window)), self.baseline_std_floor)
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

    def run_stream(self, values: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a whole stream. Because the static model is frozen after warm-up,
        scoring all post-warm-up readings in a single batch gives exactly the same
        alarms as calling `step` row by row, but is much faster.

        Returns (alarms array, mean latency per sample in ms).
        """
        values = np.asarray(values, dtype=float)
        alarms = np.zeros(len(values), dtype=int)
        t0 = time.perf_counter()
        for v in values[: self.warmup]:
            self.step({"value": float(v)})
        rest = values[self.warmup:]
        if len(rest) and self.fitted:
            scores = self._raw_score(rest.reshape(-1, 1))
            alarms[self.warmup:] = (scores > self.threshold).astype(int)
        mean_latency = (time.perf_counter() - t0) * 1000.0 / max(len(values), 1)
        return alarms, mean_latency
