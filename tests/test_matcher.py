from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.alignment.matcher import BaselineMatcher


def test_advances_only_when_score_reaches_threshold():
    matcher = BaselineMatcher(["welcome everyone", "today we explore"], threshold=80)

    chunk, score, advanced = matcher.update("unrelated words")

    assert chunk == "welcome everyone"
    assert score < 80
    assert not advanced
    assert matcher.current_index == 0


def test_advances_to_the_next_chunk_after_a_match():
    matcher = BaselineMatcher(["welcome everyone", "today we explore"], threshold=80)

    chunk, score, advanced = matcher.update("everyone welcome")

    assert score == 100
    assert advanced
    assert chunk == "today we explore"
    assert matcher.current_index == 1
    assert not matcher.finished


def test_accumulates_multiple_asr_windows_before_matching():
    matcher = BaselineMatcher(["welcome everyone", "today we explore"], threshold=80)

    _, first_score, first_advanced = matcher.update("welcome")
    chunk, second_score, second_advanced = matcher.update("everyone")

    assert first_score < 80
    assert not first_advanced
    assert second_score == 100
    assert second_advanced
    assert chunk == "today we explore"


def test_clears_transcript_history_after_advancing():
    matcher = BaselineMatcher(["welcome everyone", "today we explore"], threshold=80)
    matcher.update("welcome")
    matcher.update("everyone")

    chunk, score, advanced = matcher.update("today")

    assert chunk == "today we explore"
    assert score < 80
    assert not advanced


def test_matching_the_final_chunk_marks_the_script_complete():
    matcher = BaselineMatcher(["welcome everyone", "today we explore"], threshold=80)
    matcher.update("welcome everyone")

    chunk, score, advanced = matcher.update("today explore we")

    assert score == 100
    assert chunk == "today we explore"
    assert not advanced
    assert matcher.finished


def test_empty_script_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        BaselineMatcher([])


def test_history_window_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        BaselineMatcher(["welcome"], history_windows=0)
