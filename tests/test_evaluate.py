from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.asr.transcriber import WindowTranscript
from src.evaluation.evaluate import Bucket, WindowResult, compute_latencies, compute_metrics, replay
from src.evaluation.ground_truth import GroundTruthPointer


class _StubMatcher:
    """Advances exactly on the given 1-indexed call numbers, regardless
    of transcript content - lets tests dictate the matcher's behaviour
    directly instead of depending on real lexical/semantic scoring."""

    def __init__(self, advance_at_calls: set[int]):
        self.current_index = 0
        self._call = 0
        self._advance_at_calls = advance_at_calls

    def update(self, transcript, window_time, asr_confidence):
        self._call += 1
        advanced = self._call in self._advance_at_calls
        if advanced:
            self.current_index += 1
        return "", 0.0, advanced


def _windows(times: list[float]) -> list[WindowTranscript]:
    return [WindowTranscript(text="x", window_time=t, confidence=1.0) for t in times]


TRANSITIONS = {"0": 0.0, "1": 2.0, "2": 4.0}


def test_replay_matches_ground_truth_when_matcher_advances_on_time():
    windows = _windows([1.0, 2.0, 3.0, 4.0, 5.0])
    matcher = _StubMatcher(advance_at_calls={2, 4})
    pointer = GroundTruthPointer(TRANSITIONS)

    results = replay(windows, matcher, pointer)

    assert [r.pred_index for r in results] == [0, 1, 1, 2, 2]
    assert [r.gt_index for r in results] == [0, 1, 1, 2, 2]
    assert [r.gt_advance for r in results] == [False, True, False, True, False]
    assert [r.pred_advance for r in results] == [False, True, False, True, False]


def test_replay_flags_position_mismatch_while_matcher_lags():
    windows = _windows([1.0, 2.0, 3.0, 4.0, 5.0])
    matcher = _StubMatcher(advance_at_calls={3, 4})  # late for the t=2.0 transition
    pointer = GroundTruthPointer(TRANSITIONS)

    results = replay(windows, matcher, pointer)

    assert results[1].gt_index == 1 and results[1].pred_index == 0  # mismatch at t=2.0
    assert results[2].gt_index == 1 and results[2].pred_index == 1  # caught up at t=3.0


def test_compute_latencies_measures_seconds_to_catch_up():
    results = [
        WindowResult(window_time=1.0, gt_index=0, gt_advance=False, pred_index=0, pred_advance=False),
        WindowResult(window_time=2.0, gt_index=1, gt_advance=True, pred_index=0, pred_advance=False),
        WindowResult(window_time=3.0, gt_index=1, gt_advance=False, pred_index=1, pred_advance=True),
    ]

    assert compute_latencies(results) == pytest.approx([1.0])


def test_compute_latencies_skips_transitions_never_caught_up():
    results = [
        WindowResult(window_time=2.0, gt_index=1, gt_advance=True, pred_index=0, pred_advance=False),
        WindowResult(window_time=3.0, gt_index=1, gt_advance=False, pred_index=0, pred_advance=False),
    ]

    assert compute_latencies(results) == []


def test_compute_metrics_perfect_match():
    windows = _windows([1.0, 2.0, 3.0, 4.0, 5.0])
    matcher = _StubMatcher(advance_at_calls={2, 4})
    pointer = GroundTruthPointer(TRANSITIONS)
    results = replay(windows, matcher, pointer)
    bucket = Bucket()
    bucket.add_recording(results)

    metrics = compute_metrics(bucket)

    assert metrics.accuracy == 1.0
    assert metrics.false_advance_rate == 0.0
    assert metrics.f1_advance == 1.0
    assert metrics.n_transitions == 2
    assert metrics.n_latencies_measured == 2
    assert metrics.mean_advance_latency == pytest.approx(0.0)


def test_compute_metrics_counts_a_false_advance():
    # Advances one call early (call 1, t=1.0) with no ground-truth transition there.
    windows = _windows([1.0, 2.0, 3.0, 4.0, 5.0])
    matcher = _StubMatcher(advance_at_calls={1, 2, 4})
    pointer = GroundTruthPointer(TRANSITIONS)
    results = replay(windows, matcher, pointer)
    bucket = Bucket()
    bucket.add_recording(results)

    metrics = compute_metrics(bucket)

    assert metrics.false_advance_rate > 0.0
    assert metrics.accuracy < 1.0


def test_bucket_keeps_latency_queues_separate_per_recording():
    # Recording 1 never catches up to its one transition (target index 5,
    # way out of reach) - if its leftover pending entry leaked into
    # recording 2's queue, it would block recording 2's own transition
    # from ever resolving (FIFO) or, worse, pair up with the wrong clock.
    recording_1 = [
        WindowResult(window_time=1.0, gt_index=0, gt_advance=False, pred_index=0, pred_advance=False),
        WindowResult(window_time=2.0, gt_index=5, gt_advance=True, pred_index=0, pred_advance=False),
    ]
    recording_2 = [
        WindowResult(window_time=1.0, gt_index=0, gt_advance=False, pred_index=0, pred_advance=False),
        WindowResult(window_time=2.0, gt_index=1, gt_advance=True, pred_index=0, pred_advance=False),
        WindowResult(window_time=3.0, gt_index=1, gt_advance=False, pred_index=1, pred_advance=True),
    ]

    bucket = Bucket()
    bucket.add_recording(recording_1)
    bucket.add_recording(recording_2)

    assert bucket.latencies == pytest.approx([1.0])
    assert all(latency >= 0 for latency in bucket.latencies)


def test_bucket_merged_with_combines_windows_and_latencies():
    a = Bucket()
    a.add_recording([
        WindowResult(window_time=1.0, gt_index=1, gt_advance=True, pred_index=1, pred_advance=True),
    ])
    b = Bucket()
    b.add_recording([
        WindowResult(window_time=1.0, gt_index=1, gt_advance=True, pred_index=1, pred_advance=True),
    ])

    merged = a.merged_with([b])

    assert len(merged.windows) == 2
    assert merged.latencies == pytest.approx([0.0, 0.0])
