"""Pure conservative temporal tracking for AI detection boxes."""

from dataclasses import dataclass
import math
from typing import Iterable

from ai_targeting import (
    AimSettings,
    Detection,
    DetectionAnalysis,
    DetectionFrameSnapshot,
    TargetSnapshot,
    detection_aim_point,
)


AMBIGUITY_MARGIN = 0.15
MIN_PLAUSIBILITY_RADIUS_PX = 48.0
BOX_DIAGONAL_RADIUS_SCALE = 1.5
MIN_AREA_RATIO = 0.4
MAX_AREA_RATIO = 2.5
MAX_VELOCITY_PPS = 800.0
MAX_PREDICTION_S = 0.100
HOLD_EXPIRY_S = 0.150
STABLE_DISPLACEMENT_PX = 18.0
RECOVERY_CONFIRMATION_COUNT = 2
REPLACEMENT_CONFIRMATION_COUNT = 3
FRAME_CENTER = (160.0, 160.0)
FRAME_CENTER_TO_CORNER = math.hypot(*FRAME_CENTER)


@dataclass(frozen=True)
class _Candidate:
    index: int
    detection: Detection
    target: TargetSnapshot


@dataclass(frozen=True)
class TrackerState:
    confirmed_detection: Detection | None = None
    confirmed_target: TargetSnapshot | None = None
    preceding_target: TargetSnapshot | None = None
    last_clear_at: float | None = None
    recovery_required: bool = False
    recovery_candidate: TargetSnapshot | None = None
    recovery_count: int = 0
    pending_candidate: TargetSnapshot | None = None
    pending_count: int = 0


@dataclass(frozen=True)
class TrackingObservation:
    state: TrackerState
    analysis: DetectionAnalysis
    stability_target: TargetSnapshot | None


def _box_area(detection: Detection) -> float:
    width = detection.x2 - detection.x1
    height = detection.y2 - detection.y1
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def _accepted_detections(
    detections: Iterable[Detection],
    settings: AimSettings,
) -> tuple[Detection, ...]:
    accepted = []
    for detection in detections:
        values = (
            detection.x1,
            detection.y1,
            detection.x2,
            detection.y2,
            detection.confidence,
        )
        try:
            finite = all(math.isfinite(value) for value in values)
        except (TypeError, ValueError):
            finite = False
        if not finite or detection.confidence < settings.confidence:
            continue
        if detection_aim_point(detection) is None:
            continue
        if _box_area(detection) <= 0.0:
            continue
        accepted.append(detection)
    return tuple(accepted)


def _candidates(
    accepted: tuple[Detection, ...],
    *,
    sequence: int,
    captured_at: float,
) -> tuple[_Candidate, ...]:
    result = []
    for index, detection in enumerate(accepted):
        point = detection_aim_point(detection)
        if point is None:
            continue
        target_class, aim_x, aim_y = point
        result.append(
            _Candidate(
                index,
                detection,
                TargetSnapshot(
                    sequence,
                    captured_at,
                    target_class,
                    aim_x,
                    aim_y,
                ),
            )
        )
    return tuple(result)


def _candidate_tie_key(candidate: _Candidate) -> tuple[float, ...]:
    detection = candidate.detection
    target = candidate.target
    return (
        target.aim_x,
        target.aim_y,
        detection.x1,
        detection.y1,
        detection.x2,
        detection.y2,
        candidate.index,
    )


def _same_path(
    first: TargetSnapshot | None,
    second: TargetSnapshot,
    radius: float = STABLE_DISPLACEMENT_PX,
) -> bool:
    return (
        first is not None
        and first.target_class == second.target_class
        and math.hypot(first.aim_x - second.aim_x, first.aim_y - second.aim_y)
        <= radius
    )


def _initial_rank(
    candidates: tuple[_Candidate, ...],
) -> tuple[tuple[float, _Candidate], ...]:
    heads = tuple(
        candidate
        for candidate in candidates
        if candidate.target.target_class == "head"
    )
    preferred = heads or tuple(
        candidate
        for candidate in candidates
        if candidate.target.target_class == "player"
    )
    return tuple(
        sorted(
            (
                (
                    math.hypot(
                        candidate.target.aim_x - FRAME_CENTER[0],
                        candidate.target.aim_y - FRAME_CENTER[1],
                    )
                    / FRAME_CENTER_TO_CORNER,
                    candidate,
                )
                for candidate in preferred
            ),
            key=lambda item: (item[0], *_candidate_tie_key(item[1])),
        )
    )


def _is_ambiguous(ranked: tuple[tuple[float, _Candidate], ...]) -> bool:
    return (
        len(ranked) > 1
        and ranked[1][0] - ranked[0][0] <= AMBIGUITY_MARGIN
    )


def _intersection_over_union(first: Detection, second: Detection) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union > 0.0 else 0.0


def _predicted_point(state: TrackerState, captured_at: float) -> tuple[float, float]:
    confirmed = state.confirmed_target
    preceding = state.preceding_target
    if confirmed is None or preceding is None:
        if confirmed is None:
            return FRAME_CENTER
        return confirmed.aim_x, confirmed.aim_y
    sample_dt = confirmed.captured_at - preceding.captured_at
    if sample_dt <= 0.0:
        return confirmed.aim_x, confirmed.aim_y
    velocity_x = (confirmed.aim_x - preceding.aim_x) / sample_dt
    velocity_y = (confirmed.aim_y - preceding.aim_y) / sample_dt
    speed = math.hypot(velocity_x, velocity_y)
    if speed > MAX_VELOCITY_PPS:
        scale = MAX_VELOCITY_PPS / speed
        velocity_x *= scale
        velocity_y *= scale
    horizon = max(0.0, min(MAX_PREDICTION_S, captured_at - confirmed.captured_at))
    return (
        confirmed.aim_x + velocity_x * horizon,
        confirmed.aim_y + velocity_y * horizon,
    )


def _tracked_rank(
    state: TrackerState,
    candidates: tuple[_Candidate, ...],
    captured_at: float,
) -> tuple[tuple[float, _Candidate], ...]:
    confirmed_detection = state.confirmed_detection
    confirmed_target = state.confirmed_target
    if confirmed_detection is None or confirmed_target is None:
        return ()
    confirmed_area = _box_area(confirmed_detection)
    diagonal = math.hypot(
        confirmed_detection.x2 - confirmed_detection.x1,
        confirmed_detection.y2 - confirmed_detection.y1,
    )
    plausibility_radius = max(
        MIN_PLAUSIBILITY_RADIUS_PX,
        BOX_DIAGONAL_RADIUS_SCALE * diagonal,
    )
    predicted_x, predicted_y = _predicted_point(state, captured_at)
    ranked = []
    for candidate in candidates:
        if candidate.target.target_class != confirmed_target.target_class:
            continue
        area_ratio = _box_area(candidate.detection) / confirmed_area
        if not MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO:
            continue
        distance = math.hypot(
            candidate.target.aim_x - predicted_x,
            candidate.target.aim_y - predicted_y,
        )
        if distance > plausibility_radius:
            continue
        iou = _intersection_over_union(confirmed_detection, candidate.detection)
        score = (
            0.60 * (distance / plausibility_radius)
            + 0.25 * (1.0 - iou)
            + 0.15 * min(1.0, abs(math.log(area_ratio)))
        )
        ranked.append((score, candidate))
    return tuple(
        sorted(
            ranked,
            key=lambda item: (item[0], *_candidate_tie_key(item[1])),
        )
    )


def _confirmed_state(
    candidate: _Candidate,
    *,
    preceding: TargetSnapshot | None,
    last_clear_at: float,
    recovery_required: bool = False,
) -> TrackerState:
    return TrackerState(
        confirmed_detection=candidate.detection,
        confirmed_target=candidate.target,
        preceding_target=preceding,
        last_clear_at=last_clear_at,
        recovery_required=recovery_required,
    )


def _analysis(
    accepted: tuple[Detection, ...],
    candidate: _Candidate | None,
    *,
    sequence: int,
    captured_at: float,
) -> DetectionAnalysis:
    return DetectionAnalysis(
        target=None if candidate is None else candidate.target,
        frame=DetectionFrameSnapshot(
            sequence,
            captured_at,
            accepted,
            None if candidate is None else candidate.index,
        ),
    )


def _without_publication(
    state: TrackerState,
    accepted: tuple[Detection, ...],
    *,
    sequence: int,
    captured_at: float,
    stability_target: TargetSnapshot | None = None,
) -> TrackingObservation:
    return TrackingObservation(
        state,
        _analysis(
            accepted,
            None,
            sequence=sequence,
            captured_at=captured_at,
        ),
        stability_target,
    )


def _observe_initial(
    state: TrackerState,
    candidates: tuple[_Candidate, ...],
    accepted: tuple[Detection, ...],
    *,
    sequence: int,
    captured_at: float,
) -> TrackingObservation:
    ranked = _initial_rank(candidates)
    if not ranked:
        return _without_publication(
            TrackerState(),
            accepted,
            sequence=sequence,
            captured_at=captured_at,
        )
    best = ranked[0][1]
    ambiguous = _is_ambiguous(ranked)
    if state.pending_candidate is None and not ambiguous:
        confirmed = _confirmed_state(
            best,
            preceding=None,
            last_clear_at=captured_at,
        )
        return TrackingObservation(
            confirmed,
            _analysis(
                accepted,
                best,
                sequence=sequence,
                captured_at=captured_at,
            ),
            best.target,
        )

    pending_count = (
        state.pending_count + 1
        if _same_path(state.pending_candidate, best.target)
        else 1
    )
    if not ambiguous and pending_count >= REPLACEMENT_CONFIRMATION_COUNT:
        confirmed = _confirmed_state(
            best,
            preceding=None,
            last_clear_at=captured_at,
        )
        return TrackingObservation(
            confirmed,
            _analysis(
                accepted,
                best,
                sequence=sequence,
                captured_at=captured_at,
            ),
            best.target,
        )
    pending_state = TrackerState(
        pending_candidate=best.target,
        pending_count=min(pending_count, REPLACEMENT_CONFIRMATION_COUNT),
    )
    return _without_publication(
        pending_state,
        accepted,
        sequence=sequence,
        captured_at=captured_at,
        stability_target=None if ambiguous else best.target,
    )


def _observe_replacement(
    state: TrackerState,
    candidates: tuple[_Candidate, ...],
    accepted: tuple[Detection, ...],
    *,
    sequence: int,
    captured_at: float,
) -> TrackingObservation:
    ranked = _initial_rank(candidates)
    if not ranked or _is_ambiguous(ranked):
        reset = TrackerState(
            confirmed_detection=state.confirmed_detection,
            confirmed_target=state.confirmed_target,
            preceding_target=state.preceding_target,
            last_clear_at=state.last_clear_at,
            recovery_required=state.recovery_required,
        )
        return _without_publication(
            reset,
            accepted,
            sequence=sequence,
            captured_at=captured_at,
        )
    best = ranked[0][1]
    pending_count = (
        state.pending_count + 1
        if _same_path(state.pending_candidate, best.target)
        else 1
    )
    if pending_count >= REPLACEMENT_CONFIRMATION_COUNT:
        confirmed = _confirmed_state(
            best,
            preceding=None,
            last_clear_at=captured_at,
        )
        return TrackingObservation(
            confirmed,
            _analysis(
                accepted,
                best,
                sequence=sequence,
                captured_at=captured_at,
            ),
            best.target,
        )
    pending = TrackerState(
        confirmed_detection=state.confirmed_detection,
        confirmed_target=state.confirmed_target,
        preceding_target=state.preceding_target,
        last_clear_at=state.last_clear_at,
        recovery_required=state.recovery_required,
        pending_candidate=best.target,
        pending_count=pending_count,
    )
    return _without_publication(
        pending,
        accepted,
        sequence=sequence,
        captured_at=captured_at,
        stability_target=best.target,
    )


def observe_detections(
    state: TrackerState,
    detections: Iterable[Detection],
    settings: AimSettings,
    *,
    sequence: int,
    captured_at: float,
) -> TrackingObservation:
    """Observe one base-detection frame without mutating prior tracker state."""
    accepted = _accepted_detections(detections, settings)
    candidates = _candidates(
        accepted,
        sequence=sequence,
        captured_at=captured_at,
    )
    if state.confirmed_target is None or state.confirmed_detection is None:
        return _observe_initial(
            state,
            candidates,
            accepted,
            sequence=sequence,
            captured_at=captured_at,
        )

    last_clear_at = state.last_clear_at
    expired = last_clear_at is None or captured_at - last_clear_at >= HOLD_EXPIRY_S
    if expired:
        return _observe_replacement(
            state,
            candidates,
            accepted,
            sequence=sequence,
            captured_at=captured_at,
        )

    ranked = _tracked_rank(state, candidates, captured_at)
    if not ranked:
        held = TrackerState(
            confirmed_detection=state.confirmed_detection,
            confirmed_target=state.confirmed_target,
            preceding_target=state.preceding_target,
            last_clear_at=state.last_clear_at,
            recovery_required=state.recovery_required,
        )
        return _without_publication(
            held,
            accepted,
            sequence=sequence,
            captured_at=captured_at,
        )

    if _is_ambiguous(ranked):
        ambiguous = TrackerState(
            confirmed_detection=state.confirmed_detection,
            confirmed_target=state.confirmed_target,
            preceding_target=state.preceding_target,
            last_clear_at=state.last_clear_at,
            recovery_required=True,
        )
        return _without_publication(
            ambiguous,
            accepted,
            sequence=sequence,
            captured_at=captured_at,
        )

    best = ranked[0][1]
    if state.recovery_required:
        recovery_count = (
            state.recovery_count + 1
            if _same_path(state.recovery_candidate, best.target)
            else 1
        )
        if recovery_count < RECOVERY_CONFIRMATION_COUNT:
            recovering = TrackerState(
                confirmed_detection=state.confirmed_detection,
                confirmed_target=state.confirmed_target,
                preceding_target=state.preceding_target,
                last_clear_at=captured_at,
                recovery_required=True,
                recovery_candidate=best.target,
                recovery_count=recovery_count,
            )
            return _without_publication(
                recovering,
                accepted,
                sequence=sequence,
                captured_at=captured_at,
                stability_target=best.target,
            )

    confirmed = _confirmed_state(
        best,
        preceding=state.confirmed_target,
        last_clear_at=captured_at,
    )
    return TrackingObservation(
        confirmed,
        _analysis(
            accepted,
            best,
            sequence=sequence,
            captured_at=captured_at,
        ),
        best.target,
    )
