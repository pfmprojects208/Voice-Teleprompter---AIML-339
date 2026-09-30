from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.evaluation.ground_truth import GroundTruthPointer


def test_holds_on_the_same_chunk_before_the_next_transition():
    pointer = GroundTruthPointer({"0": 0.0, "1": 5.0, "2": 10.0})

    index, advance = pointer.update(1.0)

    assert index == 0
    assert not advance


def test_advance_fires_on_the_window_that_reaches_the_transition_and_holds_old_index():
    pointer = GroundTruthPointer({"0": 0.0, "1": 5.0, "2": 10.0})
    pointer.update(1.0)

    index, advance = pointer.update(5.0)

    assert index == 0
    assert advance


def test_current_index_moves_to_the_new_chunk_on_the_following_window():
    pointer = GroundTruthPointer({"0": 0.0, "1": 5.0, "2": 10.0})
    pointer.update(1.0)
    pointer.update(5.0)

    index, advance = pointer.update(6.0)

    assert index == 1
    assert not advance


def test_skipped_chunk_is_jumped_over_directly():
    pointer = GroundTruthPointer({"0": 0.0, "1": None, "2": 8.0})

    _, advance_before = pointer.update(3.0)
    index_at_jump, advance_at_jump = pointer.update(8.0)
    index_after, advance_after = pointer.update(9.0)

    assert not advance_before
    assert index_at_jump == 0
    assert advance_at_jump
    assert index_after == 2
    assert not advance_after


def test_unaligned_chunk_in_the_middle_is_also_jumped_over():
    pointer = GroundTruthPointer({"0": 0.0, "1": None, "2": None, "3": 12.0})

    index, advance = pointer.update(12.0)

    assert index == 0
    assert advance
    index, _ = pointer.update(13.0)
    assert index == 3


def test_rejects_transitions_with_no_timestamped_chunk():
    with pytest.raises(ValueError, match="no timestamped chunks"):
        GroundTruthPointer({"0": None, "1": None})
