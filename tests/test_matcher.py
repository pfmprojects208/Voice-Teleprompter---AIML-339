from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.alignment.matcher import BaselineMatcher, HybridMatcher


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


class _StubEmbeddingModel:
    """Deterministic stand-in for SentenceTransformer, keyed by exact text.

    Lets tests control cosine similarity directly instead of depending on a
    real model download or its non-deterministic-feeling output.
    """

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.transcript_encode_calls: list[str] = []

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            self.transcript_encode_calls.append(texts)
            return np.array(self._vectors[texts])
        return np.array([self._vectors[text] for text in texts])


HYBRID_CHUNKS = ["welcome everyone", "today we explore"]
HYBRID_CHUNK_VECTORS = {"welcome everyone": [1.0, 0.0], "today we explore": [0.0, 1.0]}


@pytest.fixture
def make_hybrid_matcher(monkeypatch):
    def _make(chunks, vectors, **kwargs):
        stub = _StubEmbeddingModel(vectors)
        monkeypatch.setattr(
            "src.alignment.matcher.SentenceTransformer", lambda model_name: stub
        )
        return HybridMatcher(chunks, **kwargs), stub

    return _make


def test_hybrid_advances_via_lexical_score_without_calling_semantic_model(make_hybrid_matcher):
    matcher, stub = make_hybrid_matcher(
        HYBRID_CHUNKS, dict(HYBRID_CHUNK_VECTORS), threshold=80, secondary_threshold=40
    )

    chunk, score, advanced = matcher.update("everyone welcome")

    assert score == 100
    assert advanced
    assert chunk == "today we explore"
    assert stub.transcript_encode_calls == []


def test_hybrid_low_lexical_score_falls_back_to_semantic_similarity(make_hybrid_matcher):
    vectors = dict(HYBRID_CHUNK_VECTORS, **{"greetings all people": [1.0, 0.0]})
    matcher, stub = make_hybrid_matcher(
        HYBRID_CHUNKS, vectors, threshold=80, secondary_threshold=40, semantic_threshold=0.9
    )

    chunk, score, advanced = matcher.update("greetings all people")

    assert advanced
    assert score == 100.0
    assert chunk == "today we explore"
    assert stub.transcript_encode_calls == ["greetings all people"]


def test_hybrid_low_lexical_and_low_semantic_score_does_not_advance(make_hybrid_matcher):
    vectors = dict(HYBRID_CHUNK_VECTORS, **{"xyz unrelated qux": [0.0, 1.0]})
    matcher, stub = make_hybrid_matcher(
        HYBRID_CHUNKS, vectors, threshold=80, secondary_threshold=40, semantic_threshold=0.5
    )

    chunk, score, advanced = matcher.update("xyz unrelated qux")

    assert not advanced
    assert score == 0.0
    assert chunk == "welcome everyone"
    assert matcher.current_index == 0


def test_hybrid_mid_range_lexical_score_does_not_trigger_semantic_fallback(make_hybrid_matcher):
    matcher, stub = make_hybrid_matcher(
        HYBRID_CHUNKS, dict(HYBRID_CHUNK_VECTORS), threshold=80, secondary_threshold=40
    )

    chunk, score, advanced = matcher.update("welcome to everyone today")

    assert not advanced
    assert 40 <= score < 80
    assert chunk == "welcome everyone"
    assert stub.transcript_encode_calls == []


def test_hybrid_matching_the_final_chunk_marks_the_script_complete(make_hybrid_matcher):
    matcher, _ = make_hybrid_matcher(
        HYBRID_CHUNKS, dict(HYBRID_CHUNK_VECTORS), threshold=80, secondary_threshold=40
    )
    matcher.update("welcome everyone")

    chunk, score, advanced = matcher.update("today explore we")

    assert score == 100
    assert chunk == "today we explore"
    assert not advanced
    assert matcher.finished


def test_hybrid_empty_script_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        HybridMatcher([])


def test_hybrid_history_window_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        HybridMatcher(["welcome"], history_windows=0)


def test_hybrid_secondary_threshold_must_be_lower_than_threshold():
    with pytest.raises(ValueError, match="secondary_threshold"):
        HybridMatcher(["welcome"], threshold=50, secondary_threshold=50)
