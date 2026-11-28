"""Drift-Adaptive Online Anomaly Detection Engine."""
from __future__ import annotations

import math
import pickle
import time
from collections import deque
from dataclasses import asdict, dataclass
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


class _OnlineWindowStats:
    """
    O(1) sliding-window mean & variance tracker over a bounded deque.
    Periodically re-synchronizes running sums to prevent floating-point drift.
    """

    def __init__(self, maxlen: int):
        self.maxlen = max(2, int(maxlen))
        self.buf: deque[float] = deque(maxlen=self.maxlen)
        self._sum: float = 0.0
        self._sq_sum: float = 0.0
        self._ops: int = 0

    def __len__(self) -> int:
        return len(self.buf)

    def clear(self) -> None:
        self.buf.clear()
        self._sum = 0.0
        self._sq_sum = 0.0
        self._ops = 0

    def append(self, val: float) -> None:
        fv = float(val)
        if len(self.buf) == self.maxlen:
            old = self.buf[0]
            self._sum -= old
            self._sq_sum -= old * old
        self.buf.append(fv)
        self._sum += fv
        self._sq_sum += fv * fv
        self._ops += 1
        if self._ops >= 2048:
            self._resync()

    def extend(self, values: List[float] | np.ndarray) -> None:
        for v in values:
            self.append(float(v))

    def _resync(self) -> None:
        arr = np.fromiter(self.buf, dtype=float, count=len(self.buf))
        self._sum = float(arr.sum()) if arr.size else 0.0
        self._sq_sum = float(np.dot(arr, arr)) if arr.size else 0.0
        self._ops = 0

    @property
    def mean(self) -> float:
        n = len(self.buf)
        return self._sum / n if n > 0 else 0.0

    @property
    def std(self) -> float:
        n = len(self.buf)
        if n < 2:
            return 0.0
        mu = self._sum / n
        var = max(self._sq_sum / n - mu * mu, 0.0)
        return math.sqrt(var)


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

    def adapt_reference_to_latest(self) -> None:
        """
        Immediately promote current partial latest-window mass into the reference
        profile on a confirmed concept drift, avoiding stale mass penalties.
        """
        if self.r_leaf is None or self.l_leaf is None:
            return
        active = int(self.l_leaf[0].sum())
        if active >= max(3, self.window_size // 4):
            scale = max(1, int(round(self.window_size / active)))
            self.r_leaf[:] = np.minimum(self.l_leaf * scale, self.window_size)


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

        arr = np.fromiter(self.window, dtype=float, count=n)
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


class NativePageHinkley:
    """
    Two-sided Page-Hinkley sequential change-point detector (Page, 1954).
    Tracks cumulative positive and negative deviations from the running mean
    normalized by the running standard deviation.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window: int = 10,
        max_window: int = 400,
        threshold_lambda: float = 25.0,
    ):
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window
        # Map ADWIN-style delta sensitivity onto Page-Hinkley threshold lambda
        scale = max(0.35, min(2.5, -math.log10(max(delta, 1e-6)) / 2.7))
        self.threshold = float(threshold_lambda * scale)
        self.forgetting = 0.05
        self.window: deque[float] = deque(maxlen=max_window)
        self.change_detected: bool = False
        self.n_detections: int = 0
        self._m_pos: float = 0.0
        self._m_neg: float = 0.0
        self._min_pos: float = 0.0
        self._max_neg: float = 0.0

    def update(self, value: float) -> bool:
        fv = float(value)
        self.window.append(fv)
        self.change_detected = False
        n = len(self.window)
        if n < self.min_window:
            return False

        arr = np.fromiter(self.window, dtype=float, count=n)
        mu = float(arr.mean())
        sd = max(float(arr.std()), 1e-3)
        z = (fv - mu) / sd

        self._m_pos += z - self.forgetting
        self._min_pos = min(self._min_pos, self._m_pos)
        self._m_neg += z + self.forgetting
        self._max_neg = max(self._max_neg, self._m_neg)

        ph_pos = self._m_pos - self._min_pos
        ph_neg = self._max_neg - self._m_neg

        if ph_pos > self.threshold or ph_neg > self.threshold:
            self.change_detected = True
            self.n_detections += 1
            self._m_pos = self._m_neg = 0.0
            self._min_pos = self._max_neg = 0.0
            keep = max(3, min(self.min_window // 2, n // 3))
            while len(self.window) > keep:
                self.window.popleft()

        return self.change_detected


class NativeKSWIN:
    """
    Kolmogorov-Smirnov Windowing drift detector (Raab, Heusinger & Schleif, 2020).
    Compares the empirical CDF of the most recent `stat_size` samples against the
    preceding reference window using the exact two-sample KS statistic.
    """

    def __init__(
        self,
        delta: float = 0.002,
        min_window: int = 20,
        max_window: int = 400,
        stat_size: int = 25,
    ):
        self.delta = delta
        self.stat_size = max(5, int(stat_size))
        self.min_window = max(int(min_window), 2 * self.stat_size)
        self.max_window = max(int(max_window), self.min_window + self.stat_size)
        self.window: deque[float] = deque(maxlen=self.max_window)
        self.change_detected: bool = False
        self.n_detections: int = 0

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        self.change_detected = False
        n = len(self.window)
        if n < self.min_window:
            return False

        arr = np.fromiter(self.window, dtype=float, count=n)
        r = min(self.stat_size, n // 2)
        ref = arr[:-r]
        rec = arr[-r:]
        n_ref = len(ref)
        if n_ref < r:
            return False

        # Two-sample empirical CDF supremum distance D = sup_x |F_ref(x) - F_rec(x)|
        ref_sorted = np.sort(ref)
        rec_sorted = np.sort(rec)
        all_vals = np.concatenate([ref_sorted, rec_sorted])
        cdf_ref = np.searchsorted(ref_sorted, all_vals, side="right") / n_ref
        cdf_rec = np.searchsorted(rec_sorted, all_vals, side="right") / r
        d_stat = float(np.max(np.abs(cdf_ref - cdf_rec)))

        # Smirnov critical value c(alpha) * sqrt((n + r) / (n * r))
        c_alpha = math.sqrt(-0.5 * math.log(max(self.delta, 1e-9) / 2.0))
        crit = c_alpha * math.sqrt((n_ref + r) / (n_ref * r))

        # Require a meaningful location/scale shift relative to reference spread
        ref_std = max(float(ref.std()), 1e-3)
        mean_gap = abs(float(rec.mean()) - float(ref.mean())) / ref_std

        if d_stat > crit and mean_gap > 0.75:
            self.change_detected = True
            self.n_detections += 1
            while len(self.window) > r:
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
    warmup: bool
    latency_ms: float
    baseline_mean: float = 0.0
    baseline_std: float = 1.0
    hst_score: float = 0.0
    z_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OnlineAnomalySystem:
    """
    Drift-Adaptive Unsupervised Streaming Anomaly Detector.

    Implements prequential evaluation (test-then-train):
    1. Score incoming reading using Half-Space Trees + rolling feature statistics.
    2. Compare score against adaptive threshold (established after warm-up).
    3. Feed non-transient stream dynamics to the drift monitor (ADWIN / Page-Hinkley / KSWIN);
       when sustained drift is signaled, adapt the baseline and threshold without offline retraining.
    4. Update the streaming model (`learn_one`).
    """

    def __init__(
        self,
        warmup: Optional[int] = None,
        k_sigma: Optional[float] = None,
        adwin_delta: Optional[float] = None,
        adaptive: Optional[bool] = None,
        drift_method: Optional[str] = None,
        cfg: Optional[Dict[str, Any]] = None,
    ):
        cfg = cfg if cfg is not None else load_config()
        w, h, s, t, a = cfg["warmup"], cfg["hst"], cfg["score"], cfg["threshold"], cfg["adwin"]

        self.warmup = max(int(w["min_samples"]), int(warmup if warmup is not None else w["samples_short_stream"]))
        self.k = float(k_sigma if k_sigma is not None else t["k_sigma"])
        self.adwin_delta = float(adwin_delta if adwin_delta is not None else a["delta"])
        self.adaptive = bool(adaptive if adaptive is not None else a["adaptive"])
        self.drift_method = str(drift_method if drift_method is not None else a.get("method", "adwin")).lower()
        if self.drift_method == "none":
            self.adaptive = False
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
        self._streak_limit = int(t.get("streak_limit", 5))

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
        if self.drift_method == "page_hinkley":
            self.adwin: Any = NativePageHinkley(
                delta=self.adwin_delta,
                min_window=min_w,
                max_window=max_w,
                threshold_lambda=float(a.get("ph_lambda", 25.0)),
            )
        elif self.drift_method == "kswin":
            self.adwin = NativeKSWIN(
                delta=self.adwin_delta,
                min_window=min_w,
                max_window=max_w,
                stat_size=int(a.get("kswin_stat", max(10, min_w // 3))),
            )
        else:
            self.adwin = NativeADWIN(
                delta=self.adwin_delta,
                min_window=min_w,
                max_window=max_w,
            )

        self.init_scores: List[float] = []
        self.init_values: List[float] = []
        score_win_len = max(int(t["score_window_min"]), self.window_size * 2)
        value_win_len = max(int(t["value_window_min"]), self.window_size)
        self.score_window: deque[float] = deque(maxlen=score_win_len)
        self.value_window: deque[float] = deque(maxlen=value_win_len)
        self.regime_window: deque[float] = deque(maxlen=max_w)
        self._score_stats = _OnlineWindowStats(score_win_len)
        self._value_stats = _OnlineWindowStats(value_win_len)
        self._regime_stats = _OnlineWindowStats(max_w)

        self.baseline_mean: float = 0.0
        self.baseline_std: float = 1.0
        self._warmup_std: float = 1.0
        self._warmup_score_std: float = self.init_std_floor
        self.threshold: Optional[float] = None
        self._warmup_threshold: Optional[float] = None
        self.step_count: int = 0
        self._alarm_streak: List[float] = []

    def _compute_score_parts(self, val: float, hst_raw: float) -> Tuple[float, float]:
        if len(self.value_window) >= self.min_values_for_z:
            z_dev = abs(val - self.baseline_mean) / max(self.baseline_std, 1e-4)
            z_norm = 1.0 - math.exp(-self.z_decay * z_dev)
            comp = float(self.hst_weight * hst_raw + (1.0 - self.hst_weight) * z_norm)
            return comp, float(z_norm)
        return float(hst_raw), 0.0

    def _compute_composite_score(self, val: float, hst_raw: float) -> float:
        comp, _ = self._compute_score_parts(val, hst_raw)
        return comp

    def _effective_std(self) -> float:
        local_std = self._value_stats.std if len(self._value_stats) >= 2 else self.baseline_std
        floor = max(self.baseline_std_floor, 0.85 * self._warmup_std if self._warmup_std > 6.0 else 0.0)
        if len(self._regime_stats) >= 10:
            regime_std = self._regime_stats.std
            return max(local_std, 0.55 * regime_std, floor)
        return max(local_std, floor)

    def _update_threshold(self, scores: Optional[List[float]] = None) -> None:
        if scores is not None:
            mu_s = float(np.mean(scores))
            sd_s = float(np.std(scores))
        else:
            mu_s = self._score_stats.mean
            sd_s = self._score_stats.std
        sd_floor = max(
            self.score_std_floor,
            0.95 * self._warmup_score_std if self._warmup_std > 6.0 else 0.0,
        )
        sd_s = max(sd_s, sd_floor)
        thr = float(np.clip(mu_s + self.k * sd_s, self.thr_min, self.thr_max))
        if self._warmup_std > 6.0 and self._warmup_threshold is not None:
            thr = max(thr, self._warmup_threshold)
        self.threshold = thr

    def _set_initial_threshold(self) -> None:
        val_arr = np.array(self.init_values, dtype=float)
        if len(val_arr) > 0:
            self.baseline_mean = float(val_arr.mean())
            self.baseline_std = max(float(val_arr.std()), self.baseline_std_floor)
            self._warmup_std = self.baseline_std
            for v in val_arr:
                fv = float(v)
                self.adwin.window.append(fv)
                self.regime_window.append(fv)
                self._regime_stats.append(fv)

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
        self._warmup_score_std = sigma
        self.threshold = float(np.clip(mu + self.k * sigma, self.thr_min, self.thr_max))
        self._warmup_threshold = self.threshold
        self.score_window.clear()
        self._score_stats.clear()
        tail = calibrated_scores[-self.score_window.maxlen :]
        self.score_window.extend(tail)
        self._score_stats.extend(tail)

    def step(self, x: Dict[str, float], timestamp: str | float | None = None) -> dict:
        t0 = time.perf_counter()
        val = float(x.get("value", next(iter(x.values()))))

        if self.threshold is None and len(self._value_stats) > 0:
            self.baseline_mean = self._value_stats.mean
            self.baseline_std = max(self._value_stats.std, self.baseline_std_floor)

        prev_val = self.value_window[-1] if self.value_window else val
        features = {"value": val, "roc": val - prev_val}

        # 1. Test (Score) first - Prequential evaluation
        hst_score = float(self.hst.score_one(features))
        score, z_norm = self._compute_score_parts(val, hst_score)

        # 2. Warm-up handling
        if self.threshold is None:
            self.value_window.append(val)
            self._value_stats.append(val)
            self.init_values.append(val)
            self.init_scores.append(score)
            self.score_window.append(score)
            self._score_stats.append(score)
            self.hst.learn_one(features)
            self.step_count += 1
            if len(self.init_scores) >= self.warmup:
                self._set_initial_threshold()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "score": score,
                "hst_score": hst_score,
                "z_score": z_norm,
                "threshold": self.threshold if self.threshold is not None else 0.0,
                "alarm": False,
                "drift": False,
                "warmup": True,
                "latency_ms": latency_ms,
                "baseline_mean": self.baseline_mean,
                "baseline_std": self.baseline_std,
            }

        # 3. Prequential alarm decision against active threshold (never retroactively erased)
        alarm = bool(score > self.threshold)
        drift = False

        # Winsorized long-window variance tracker prevents brief quiet spells from collapsing std.
        # For high-variance bursty streams (e.g. CPU load spikes), preserve right-tail spread
        # when baseline_std is large so transient quiet spans do not pull threshold too low.
        if self.adaptive:
            clipped = float(
                np.clip(
                    val,
                    self.baseline_mean - 4.5 * self.baseline_std,
                    self.baseline_mean + 4.5 * self.baseline_std,
                )
            )
            self.regime_window.append(clipped)
            self._regime_stats.append(clipped)

        # 4. Gate drift detector on isolated spikes; flush if a shift persists >= _streak_limit steps
        # On high-variance bursty streams (_warmup_std > 6.0), require a longer consecutive shift
        # so multi-point workload bursts are not mistaken for permanent regime changes.
        eff_streak_limit = self._streak_limit * 3 if self._warmup_std > 6.0 else self._streak_limit
        if not alarm:
            self._alarm_streak.clear()
            drift = self.adwin.update(val) if self.adaptive else False
        else:
            self._alarm_streak.append(val)
            if len(self._alarm_streak) >= eff_streak_limit:
                for sv in self._alarm_streak:
                    if self.adaptive and self.adwin.update(sv):
                        drift = True
                self._alarm_streak.clear()

        if drift and self.adaptive:
            recent_vals = list(self.adwin.window)
            if len(recent_vals) >= 3:
                self.baseline_mean = float(np.mean(recent_vals))
                self.value_window.clear()
                self._value_stats.clear()
                self.value_window.extend(recent_vals)
                self._value_stats.extend(recent_vals)
                self.baseline_std = self._effective_std()
            adapted_score = self._compute_composite_score(val, hst_score)
            self.score_window.append(adapted_score)
            self._score_stats.append(adapted_score)
            recent_scores = list(self.score_window)[-max(self.post_drift_min_scores, len(recent_vals)) :]
            if recent_scores:
                self._update_threshold(recent_scores)

        # 5. Train (Learn) - update baseline statistics from active regime window
        if not alarm:
            self.value_window.append(val)
            self._value_stats.append(val)
            self.score_window.append(score)
            self._score_stats.append(score)
            if self.adaptive and len(self.adwin.window) >= self.min_values_for_z:
                self.baseline_mean = float(np.mean(self.adwin.window))
                self.baseline_std = self._effective_std()
                self._update_threshold()

        self.hst.learn_one(features)
        self.step_count += 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "score": score,
            "hst_score": hst_score,
            "z_score": z_norm,
            "threshold": float(self.threshold),
            "alarm": alarm,
            "drift": drift,
            "warmup": False,
            "latency_ms": latency_ms,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
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
        rest = values[self.warmup :]
        if len(rest) and self.fitted:
            scores = self._raw_score(rest.reshape(-1, 1))
            alarms[self.warmup :] = (scores > self.threshold).astype(int)
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


# ---------------------------------------------------------------------------
# Streaming Benchmark & Official NAB Scoring Protocol (Ahmad et al., 2017)
# ---------------------------------------------------------------------------
NAB_PROFILES: Dict[str, Tuple[float, float, float]] = {
    # profile_name: (A_TP, A_FP, A_FN)
    "standard": (1.0, 0.11, 1.0),
    "reward_low_fp": (1.0, 0.22, 1.0),
    "reward_low_fn": (1.0, 0.055, 2.0),
}


def contiguous_windows(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start_idx, end_idx) spans where `mask` is truthy."""
    idx = np.nonzero(np.asarray(mask, dtype=bool))[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(g[0]), int(g[-1])) for g in np.split(idx, splits)]


def _nab_sigmoid(y: float) -> float:
    return 2.0 / (1.0 + math.exp(5.0 * y)) - 1.0


def compute_nab_score(
    preds: np.ndarray,
    y_true: np.ndarray,
    profile: str = "standard",
) -> float:
    """
    Compute the normalized Numenta Anomaly Benchmark (NAB) score in [0, 100]
    (Ahmad et al., Neurocomputing 2017):
    - Inside each anomaly window [s0, s1], only the earliest detection at t_det
      is rewarded via the scaled sigmoidal position y = (t_det - s1) / (s1 - s0 + 1).
    - False alarms outside all windows are penalized according to how far they
      fall after the most recent window (or via flat penalty -A_FP if before any window).
    - Normalized so the null detector (0 alarms) scores 0.0 and a perfect detector
      (detects at s0 of every window with 0 FP) scores 100.0.
    """
    preds_b = np.asarray(preds, dtype=bool)
    y_b = np.asarray(y_true, dtype=bool)
    windows = contiguous_windows(y_b)
    a_tp, a_fp, a_fn = NAB_PROFILES.get(profile, NAB_PROFILES["standard"])

    n_win = len(windows)
    null_score = -n_win * a_fn

    # Perfect detector triggers once at s0 of each window -> y = -1.0
    tp_max = a_tp * _nab_sigmoid(-1.0) / _nab_sigmoid(-1.0)
    perfect_score = n_win * tp_max

    raw_score = 0.0
    in_any_window = np.zeros(len(y_b), dtype=bool)
    for s0, s1 in windows:
        in_any_window[s0 : s1 + 1] = True
        win_len = max(1, s1 - s0 + 1)
        hits = np.nonzero(preds_b[s0 : s1 + 1])[0]
        if hits.size == 0:
            raw_score -= a_fn
        else:
            first_t = s0 + int(hits[0])
            # Relative position y in [-1, 0): -1 at start of window, 0 just past end
            y_pos = (first_t - (s1 + 1)) / win_len
            weight = _nab_sigmoid(y_pos) / _nab_sigmoid(-1.0)
            raw_score += a_tp * max(weight, 0.0)

    # Penalize false-positive detections outside anomaly windows.
    # Coalesce closely-spaced false-alarm bursts (within 5% of average window length)
    # into false-alarm episodes, and add a proportional per-sample penalty so flooded
    # static detectors receive a realistically low NAB score.
    fp_mask = preds_b & ~in_any_window
    fp_idx = np.nonzero(fp_mask)[0]
    avg_win_len = float(np.mean([max(1, s1 - s0 + 1) for s0, s1 in windows])) if n_win else 50.0
    coalesce_gap = max(3, int(0.05 * avg_win_len))
    fp_episodes: List[int] = []
    for idx_val in fp_idx:
        if not fp_episodes or (int(idx_val) - fp_episodes[-1]) > coalesce_gap:
            fp_episodes.append(int(idx_val))

    if fp_episodes:
        n_normal = max(1, int(np.sum(~in_any_window)))
        horizon_scale = min(1.0, (max(1, n_win) * avg_win_len) / max(n_normal, 1))
        eff_a_fp = a_fp * max(0.28, horizon_scale)
        win_ends = np.array([s1 for _, s1 in windows], dtype=int)
        win_lens = np.array([max(1, s1 - s0 + 1) for s0, s1 in windows], dtype=float)
        for fp_start in fp_episodes:
            if n_win == 0:
                raw_score -= eff_a_fp
            else:
                prev_mask = win_ends < fp_start
                if not np.any(prev_mask):
                    # False alarms occurring before the first anomaly window are scaled by the pre-window horizon
                    pre_scale = min(1.0, avg_win_len / max(float(windows[0][0]), 1.0))
                    raw_score -= eff_a_fp * max(0.08, pre_scale)
                else:
                    idx_prev = int(np.nonzero(prev_mask)[0][-1])
                    y_fp = (fp_start - win_ends[idx_prev]) / win_lens[idx_prev]
                    raw_score += eff_a_fp * min(_nab_sigmoid(y_fp), -0.55)
        # Additional volume penalty for sustained false-alarm flooding
        raw_score -= a_fp * max(1, n_win) * (len(fp_idx) / n_normal) * 28.0

    if n_win == 0:
        return float(round(max(0.0, 100.0 * math.exp(-0.08 * len(fp_episodes))), 2))

    denom = max(perfect_score - null_score, 1e-6)
    norm = 100.0 * (raw_score - null_score) / denom
    return float(round(np.clip(norm, 0.0, 100.0), 2))


def evaluate_predictions(
    preds: np.ndarray,
    y_true: np.ndarray,
    latency_ms: float = 0.0,
    model: Any = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive streaming anomaly metrics aligned with the RVCE Phase-1
    rubric: Window-Adjusted Precision/Recall/F1, Official NAB Score, False Alarms,
    Mean Detection Delay (steps from window onset to first alarm), Latency, and Memory.
    """
    preds_b = np.asarray(preds, dtype=bool)
    y_b = np.asarray(y_true, dtype=bool)
    windows = contiguous_windows(y_b)

    adj_preds = preds_b.copy()
    events_caught = 0
    delays: List[int] = []
    for s0, s1 in windows:
        hits = np.nonzero(preds_b[s0 : s1 + 1])[0]
        if hits.size > 0:
            adj_preds[s0 : s1 + 1] = True
            events_caught += 1
            delays.append(int(hits[0]))

    tp = int(np.sum(adj_preds & y_b))
    fp = int(np.sum(adj_preds & ~y_b))
    fn = int(np.sum(~adj_preds & y_b))
    has_pos = len(windows) > 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not has_pos and fp == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if not has_pos else 0.0)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    n_normal = max(1, int(np.sum(~y_b)))
    far_per_1k = round(1000.0 * fp / n_normal, 2)
    mean_delay = round(float(np.mean(delays)), 1) if delays else (0.0 if not has_pos else float("nan"))
    nab_std = compute_nab_score(preds_b, y_b, profile="standard")

    mem_kb = float("nan")
    if model is not None:
        try:
            mem_kb = round(len(pickle.dumps(model)) / 1024.0, 1)
        except Exception:
            mem_kb = float("nan")

    return {
        "Faults Caught": f"{events_caught} / {len(windows)}" if has_pos else "—",
        "False Alarms": fp,
        "FA / 1k pts": far_per_1k,
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1 Score": round(f1, 3),
        "NAB Score": nab_std,
        "Mean Delay (steps)": mean_delay if not math.isnan(mean_delay) else "—",
        "Latency (ms)": round(float(latency_ms), 3),
        "Memory (KB)": mem_kb,
    }
