"""A deterministic stand-in for `nidra.serve.predictor.NidraPredictor`.

**Placeholder.** Every number this module produces is arithmetic over the last few
observed windows, not a learned transition. It exists so the serving plane — streams,
sequence buffers, scale-out, persistence, the WebSocket — can be built and tested before
the world model lands, and it is deleted from the runtime path the moment
`predictor.impl` is switched to `nidra`. Nothing here may be quoted as a result.

What it does reproduce faithfully is the *shape* of the contract: the exact
`NidraPredictor` interface from IMPLEMENTATION-ML.md §7, a `Forecast`-valid payload with
horizons `k = 1..K`, a band that widens with the horizon, a stage distribution that
shifts benign → recon → initial_access as risk rises, and a counterfactual labelled
"model-internal what-if". A downstream service that works against this one works against
the real predictor.

The surrogate risk is a logistic over four drivers — `syn_ratio`, `dst_port_entropy`,
`new_peer_count`, `d_out_degree` — each read at its last observed level and at its
backward-looking slope over the last three windows. Levels are squashed through `tanh`
so an unbounded count cannot dominate a ratio. Risk drifts with the horizon at a rate
set by those slopes: a host whose drivers are climbing keeps climbing, and clamping one
of them flat in `counterfactual` visibly bends the curve back down.

Nothing in here looks past the last row of `states`. That is the same rule the real
model obeys, and it is worth obeying in the placeholder so the pipeline around it is
never accidentally built on a leak.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from nidra.data.schema import FEATURE_ORDER, STAGES
from nidra_common.config import get_config

#: Written into every `Forecast` this predictor produces. Distinguishable at a glance in
#: the database and on the console from anything the trained model emits.
MODEL_VERSION = "stub-0"

#: CLAUDE.md invariant, judge-facing. The counterfactual answers a question about the
#: model, not about the network, and the response says so in as many words.
COUNTERFACTUAL_LABEL = "model-internal what-if"

#: Position of each feature in a `[L, 45]` state array. Built from `FEATURE_ORDER`, once.
FEATURE_INDEX: dict[str, int] = {name: index for index, name in enumerate(FEATURE_ORDER)}

#: The four features the surrogate risk reads. Chosen because they are the ones a
#: reconnaissance-to-intrusion trajectory actually moves.
DRIVERS: tuple[str, ...] = (
    "syn_ratio",
    "dst_port_entropy",
    "new_peer_count",
    "d_out_degree",
)

#: Divisor inside the `tanh`, roughly the value at which each driver counts as high.
#: `syn_ratio` is already a ratio; a port entropy near ln(20); ten new peers in a window;
#: a fan-out that grew by five hosts since the last one.
DRIVER_SCALE: dict[str, float] = {
    "syn_ratio": 1.0,
    "dst_port_entropy": 3.0,
    "new_peer_count": 10.0,
    "d_out_degree": 5.0,
}

#: Weight on the squashed level of each driver.
LEVEL_WEIGHT: dict[str, float] = {
    "syn_ratio": 2.4,
    "dst_port_entropy": 1.1,
    "new_peer_count": 0.35,
    "d_out_degree": 0.45,
}

#: Weight on the squashed slope of each driver. Slopes carry most of the horizon drift:
#: where a host is heading matters more here than where it currently sits.
SLOPE_WEIGHT: dict[str, float] = {
    "syn_ratio": 1.8,
    "dst_port_entropy": 1.4,
    "new_peer_count": 0.6,
    "d_out_degree": 0.3,
}

#: Logit of a host with every driver at zero — quiet, and scored as such.
INTERCEPT = -3.2

#: Per-step logit drift with flat drivers. Small and positive: uncertainty accumulates
#: over the horizon even when nothing is moving.
DRIFT_BASE = 0.12

#: How much the driver slopes steer the drift. Negative slopes bend the curve down.
DRIFT_GAIN = 0.25

#: Band half-width at k=0 and its growth per step. Linear, per the P7 brief; the real
#: model's band comes from ensemble spread and will not be linear.
CI_BASE = 0.03
CI_PER_K = 0.035

#: Windows the backward slope is fitted over. Matches `slope3_*` in the feature set.
SLOPE_WINDOW = 3

#: Constant mass held on the late stages. The surrogate has no evidence about lateral
#: movement, C2 or exfiltration, so it never claims any — but zero is also a claim.
RESIDUAL_STAGES: tuple[str, ...] = ("lateral", "c2", "exfil")
RESIDUAL_MASS = 0.01

#: Below this the slope reads as flat, and the signal is described as steady rather than
#: rising or falling.
STEADY_EPS = 1e-6

#: JSON is easier to read and diff without eighteen digits of float noise, and rounding
#: is monotone so it cannot invert `ci_low <= p_compromise <= ci_high`.
ROUND_DP = 6

#: Display copy for each driver, and the range its extrapolated value is clamped into.
DRIVER_DISPLAY: dict[str, str] = {
    "syn_ratio": "SYN ratio",
    "dst_port_entropy": "destination-port entropy",
    "new_peer_count": "new peers contacted",
    "d_out_degree": "fan-out change",
}

DRIVER_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "syn_ratio": (0.0, 1.0),
    "dst_port_entropy": (0.0, None),
    "new_peer_count": (0.0, None),
    "d_out_degree": (None, None),
}


# --------------------------------------------------------------------------- numerics


def sigmoid(z: float) -> float:
    """Logistic, written so a large negative `z` underflows to 0.0 instead of raising."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def recent_slope(series: np.ndarray, window: int = SLOPE_WINDOW) -> float:
    """OLS slope per window over the last `window` points, zero-padded below that many.

    Strictly backward-looking: the series ends at the last observed window and nothing
    after it exists. Fewer than `window` points is a sequence start, and a slope fitted
    to two points there would be an artefact of padding rather than a trend.
    """
    tail = np.asarray(series[-window:], dtype=float)
    if tail.size < window:
        return 0.0
    x = np.arange(tail.size, dtype=float)
    x_centred = x - x.mean()
    denominator = float((x_centred**2).sum())
    if denominator == 0.0:
        return 0.0
    return float((x_centred * (tail - tail.mean())).sum() / denominator)


def clamp(value: float, low: float | None, high: float | None) -> float:
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# ------------------------------------------------------------------- internal structs


@dataclass(frozen=True)
class DriverReading:
    """The last observed level and the recent slope of each driver, one per key."""

    levels: dict[str, float]
    slopes: dict[str, float]


@dataclass(frozen=True)
class TrajectoryPoint:
    """One step of the surrogate trajectory, before it is dressed as a `HorizonPoint`."""

    k: int
    p_compromise: float
    ci_low: float
    ci_high: float
    stage_dist: dict[str, float]


def validate_states(states: np.ndarray) -> np.ndarray:
    """Fail loudly on anything that is not an `[L, 45]` block of observed windows.

    Schema drift caught here costs a second. Caught downstream it costs an afternoon of
    wondering why a feature attribution names the wrong column.
    """
    array = np.asarray(states, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"states must be [L, F], got shape {array.shape}")
    if array.shape[1] != len(FEATURE_ORDER):
        raise ValueError(
            f"states has {array.shape[1]} features, FEATURE_ORDER has {len(FEATURE_ORDER)}"
        )
    if array.shape[0] < 1:
        raise ValueError("states must carry at least one observed window")
    return array


def stage_distribution(risk: float) -> dict[str, float]:
    """Mass shifting benign → recon → initial_access as `risk` rises, summing to 1.

    A quadratic Bernstein basis: benign dominates at low risk, recon peaks in the middle
    where a host is fanning out but has not landed anywhere, initial_access takes over as
    risk approaches one. The late stages hold a constant sliver — the surrogate has no
    evidence for them, and pretending to certainty about their absence would be a claim.
    """
    residual = RESIDUAL_MASS * len(RESIDUAL_STAGES)
    head = 1.0 - residual
    dist = {
        "benign": head * (1.0 - risk) ** 2,
        "recon": head * 2.0 * risk * (1.0 - risk),
        "initial_access": head * risk**2,
    }
    dist.update(dict.fromkeys(RESIDUAL_STAGES, RESIDUAL_MASS))
    return {stage: round(dist[stage], ROUND_DP) for stage in STAGES}


class StubPredictor:
    """`NidraPredictor`'s interface, backed by arithmetic instead of a world model.

    **Placeholder until `nidra.serve.predictor` lands.** Deterministic — the same states
    give the same forecast on every call, in every worker — and fast enough that the
    300 ms serving target is met by three orders of magnitude, which is exactly the point
    while the rest of the plane is being built.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg if cfg is not None else get_config()

    def __repr__(self) -> str:
        return f"StubPredictor(model_version={MODEL_VERSION!r})"

    # ------------------------------------------------------------------ configuration

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    @property
    def window_delta(self) -> int:
        return int(self.cfg["window_delta"])

    @property
    def horizon_k(self) -> int:
        return int(self.cfg["horizon_K"])

    @property
    def risk_threshold(self) -> float:
        return float(self.cfg["risk_threshold"])

    @property
    def lead_time_m(self) -> int:
        return int(self.cfg["lead_time_m"])

    # ---------------------------------------------------------------- the interface

    def forecast(self, states: np.ndarray, host_id: str, origin_ts: datetime) -> dict[str, Any]:
        """A K-step trajectory for one host, anchored at `origin_ts`.

        `states` is `[L, 45]` raw observed windows, oldest first, in `FEATURE_ORDER`.
        The returned dict is `Forecast`-shaped bar `tenant_id`, which the caller owns —
        the ML interface is deliberately tenant-blind.
        """
        array = validate_states(states)
        drivers = self._read_drivers(array)
        points = self._trajectory(drivers)
        origin = _as_utc(origin_ts)
        observed_risk = round(sigmoid(self._base_logit(drivers)), ROUND_DP)

        horizons = [
            {
                "k": point.k,
                "ts": origin + timedelta(seconds=point.k * self.window_delta),
                "p_compromise": point.p_compromise,
                "ci_low": point.ci_low,
                "ci_high": point.ci_high,
                "stage_dist": point.stage_dist,
                "predicted_features": self._predicted_features(array, drivers, point.k),
            }
            for point in points
        ]

        return {
            "host_id": host_id,
            "origin_ts": origin,
            "horizons": horizons,
            "lead_time_s": self._lead_time_s(points),
            "observed_stage": self._argmax_stage(stage_distribution(observed_risk)),
            "observed_risk": observed_risk,
            "top_signals": self._top_signals(drivers),
            "driving_window": self._driving_window(array),
            "model_version": MODEL_VERSION,
        }

    def counterfactual(
        self, states: np.ndarray, feature_name: str, clamp_value: float
    ) -> dict[str, Any]:
        """Re-simulate with one feature held at `clamp_value` in every observed window.

        A question about the model, not about the network — hence the label. It says what
        this predictor would have projected on a different input, and nothing at all about
        what the network would have done.
        """
        array = validate_states(states)
        if feature_name not in FEATURE_INDEX:
            raise ValueError(f"unknown feature {feature_name!r}; not in FEATURE_ORDER")

        clamped = array.copy()
        clamped[:, FEATURE_INDEX[feature_name]] = float(clamp_value)

        return {
            "label": COUNTERFACTUAL_LABEL,
            "feature": feature_name,
            "clamp_value": float(clamp_value),
            "model_version": MODEL_VERSION,
            "original": [self._curve_point(point) for point in self._trajectory_of(array)],
            "counterfactual": [self._curve_point(point) for point in self._trajectory_of(clamped)],
        }

    def explain(self, states: np.ndarray, horizon_k: int) -> dict[str, Any]:
        """Attribution for the step-`horizon_k` projection: signals plus window weights.

        The real predictor separates SHAP on the risk head from input-gradient saliency
        over past windows (IMPLEMENTATION-ML.md §6). The surrogate has one mechanism and
        says so in `method`, so nothing downstream mistakes it for an explanation.
        """
        array = validate_states(states)
        if not 1 <= horizon_k <= self.horizon_k:
            raise ValueError(f"horizon_k must be in 1..{self.horizon_k}, got {horizon_k}")

        return {
            "horizon_k": horizon_k,
            "method": "stub-surrogate",
            "model_version": MODEL_VERSION,
            "top_signals": self._top_signals(self._read_drivers(array)),
            "driving_window": self._driving_window(array),
            "window_importance": self._window_importance(array),
            "note": "placeholder attributions; not SHAP, not saliency",
        }

    # ------------------------------------------------------------------ surrogate risk

    def _read_drivers(self, states: np.ndarray) -> DriverReading:
        """Last observed level and backward slope of each driver."""
        levels: dict[str, float] = {}
        slopes: dict[str, float] = {}
        for name in DRIVERS:
            column = states[:, FEATURE_INDEX[name]]
            levels[name] = float(column[-1])
            slopes[name] = recent_slope(column)
        return DriverReading(levels=levels, slopes=slopes)

    def _contributions(self, drivers: DriverReading) -> dict[str, float]:
        """Each driver's signed share of the logit — level and slope together."""
        return {
            name: LEVEL_WEIGHT[name] * math.tanh(drivers.levels[name] / DRIVER_SCALE[name])
            + SLOPE_WEIGHT[name] * math.tanh(drivers.slopes[name] / DRIVER_SCALE[name])
            for name in DRIVERS
        }

    def _base_logit(self, drivers: DriverReading) -> float:
        """Logit of the *observed* state, before any horizon drift."""
        return INTERCEPT + sum(self._contributions(drivers).values())

    def _drift(self, drivers: DriverReading) -> float:
        """Logit change per horizon step. Rising drivers steepen it; falling ones bend it."""
        slope_term = sum(
            SLOPE_WEIGHT[name] * math.tanh(drivers.slopes[name] / DRIVER_SCALE[name])
            for name in DRIVERS
        )
        return DRIFT_BASE + DRIFT_GAIN * slope_term

    def _trajectory(self, drivers: DriverReading) -> list[TrajectoryPoint]:
        """Steps `k = 1..K`, risk drifting with the horizon and the band widening with it."""
        base = self._base_logit(drivers)
        drift = self._drift(drivers)
        points: list[TrajectoryPoint] = []
        for k in range(1, self.horizon_k + 1):
            risk = sigmoid(base + k * drift)
            half = CI_BASE + CI_PER_K * k
            points.append(
                TrajectoryPoint(
                    k=k,
                    p_compromise=round(risk, ROUND_DP),
                    ci_low=round(max(0.0, risk - half), ROUND_DP),
                    ci_high=round(min(1.0, risk + half), ROUND_DP),
                    stage_dist=stage_distribution(risk),
                )
            )
        return points

    def _trajectory_of(self, states: np.ndarray) -> list[TrajectoryPoint]:
        return self._trajectory(self._read_drivers(states))

    @staticmethod
    def _curve_point(point: TrajectoryPoint) -> dict[str, float]:
        return {
            "k": point.k,
            "p_compromise": point.p_compromise,
            "ci_low": point.ci_low,
            "ci_high": point.ci_high,
        }

    # -------------------------------------------------------------------- derived copy

    def _lead_time_s(self, points: list[TrajectoryPoint]) -> float | None:
        """Seconds from `origin_ts` to the first sustained crossing of the threshold.

        Sustained means `lead_time_m` consecutive windows at or above it — one window
        over the line is noise, and a lead-time badge built on noise is worse than none.
        `None` when the trajectory never crosses inside the horizon.
        """
        threshold = self.risk_threshold
        run = self.lead_time_m
        for index in range(len(points) - run + 1):
            window = points[index : index + run]
            if all(point.p_compromise >= threshold for point in window):
                return float(window[0].k * self.window_delta)
        return None

    @staticmethod
    def _argmax_stage(stage_dist: dict[str, float]) -> str:
        """Most likely stage; ties broken by `STAGES` order, which is the escalation order."""
        return max(STAGES, key=lambda stage: stage_dist.get(stage, 0.0))

    def _top_signals(self, drivers: DriverReading) -> list[dict[str, Any]]:
        """The drivers behind this forecast, largest absolute contribution first."""
        contributions = self._contributions(drivers)
        signals = []
        for name in DRIVERS:
            slope = drivers.slopes[name]
            direction = "up" if slope >= 0 else "down"
            signals.append(
                {
                    "name": name,
                    "shap_value": round(contributions[name], ROUND_DP),
                    "direction": direction,
                    "display": f"{DRIVER_DISPLAY[name]} {self._trend_word(slope)}",
                }
            )
        signals.sort(key=lambda signal: abs(signal["shap_value"]), reverse=True)
        return signals

    @staticmethod
    def _trend_word(slope: float) -> str:
        if slope > STEADY_EPS:
            return "rising"
        if slope < -STEADY_EPS:
            return "falling"
        return "steady"

    def _window_importance(self, states: np.ndarray) -> list[float]:
        """Per-window weight: how much the drivers moved entering that window, normalised.

        Index 0 is the oldest window of the context. The first window has no predecessor
        inside the context, so its movement is zero rather than a difference against a
        window nobody observed.
        """
        weights = np.zeros(states.shape[0], dtype=float)
        for name in DRIVERS:
            column = states[:, FEATURE_INDEX[name]]
            weights[1:] += np.abs(np.diff(column)) / DRIVER_SCALE[name]
        total = float(weights.sum())
        if total == 0.0:
            uniform = 1.0 / weights.size
            return [round(uniform, ROUND_DP)] * weights.size
        return [round(float(weight / total), ROUND_DP) for weight in weights]

    def _driving_window(self, states: np.ndarray) -> int:
        """Index into the context of the window the drivers moved most on entering.

        0 is the oldest window, `L - 1` the most recent. Ties resolve to the later
        window: recent movement is the more useful thing to point a console at.
        """
        weights = self._window_importance(states)
        best = 0
        for index, weight in enumerate(weights):
            if weight >= weights[best]:
                best = index
        return best

    def _predicted_features(
        self, states: np.ndarray, drivers: DriverReading, k: int
    ) -> dict[str, float]:
        """Last observed window with the four drivers carried forward along their slopes.

        Everything else is held at its last observed value. The real transition model
        decodes all 45; this one moves the features it actually reasons about and leaves
        the rest visibly unchanged rather than inventing a trajectory for them.
        """
        projected = {
            name: round(float(states[-1, index]), ROUND_DP) for name, index in FEATURE_INDEX.items()
        }
        for name in DRIVERS:
            low, high = DRIVER_BOUNDS[name]
            value = drivers.levels[name] + k * drivers.slopes[name]
            projected[name] = round(clamp(value, low, high), ROUND_DP)
        return projected
