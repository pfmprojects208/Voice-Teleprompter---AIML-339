from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.evaluation.features import FeatureExtractor


class _StubEmbeddingModel:
    """Deterministic stand-in for SentenceTransformer, keyed by exact text."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors
        self.encode_calls: list[str] = []

    def encode(self, texts, convert_to_tensor=True):
        if isinstance(texts, str):
            self.encode_calls.append(texts)
            return np.array(self._vectors[texts])
        return np.array([self._vectors[text] for text in texts])


CHUNKS = ["welcome everyone", "today we explore", "thanks for watching"]
CHUNK_VECTORS = {
    "welcome everyone": [1.0, 0.0, 0.0],
    "today we explore": [0.0, 1.0, 0.0],
    "thanks for watching": [0.0, 0.0, 1.0],
}


@pytest.fixture
def make_extractor(monkeypatch):
    def _make(chunks, vectors, **kwargs):
        stub = _StubEmbeddingModel(vectors)
        monkeypatch.setattr("src.evaluation.features.SentenceTransformer", lambda model_name: stub)
        return FeatureExtractor(chunks, **kwargs), stub

    return _make


def test_lexical_and_semantic_current_match_the_active_chunk(make_extractor):
    extractor, _ = make_extractor(CHUNKS, dict(CHUNK_VECTORS, **{"welcome everyone": [1.0, 0.0, 0.0]}))

    features = extractor.update("welcome everyone", current_index=0, window_time=1.0, asr_confidence=0.9)

    assert features.lexical_current == 100
    assert features.semantic_current == pytest.approx(1.0)
    assert features.asr_confidence == 0.9


def test_semantic_next_and_delta_compare_against_the_following_chunk(make_extractor):
    vectors = dict(CHUNK_VECTORS, **{"today explore we": [0.0, 1.0, 0.0]})
    extractor, _ = make_extractor(CHUNKS, vectors)

    features = extractor.update("today explore we", current_index=0, window_time=1.0, asr_confidence=0.9)

    assert features.semantic_current == pytest.approx(0.0)
    assert features.semantic_next == pytest.approx(1.0)
    assert features.semantic_delta == pytest.approx(1.0)


def test_last_chunk_has_no_next_so_delta_is_zero(make_extractor):
    vectors = dict(CHUNK_VECTORS, **{"thanks for watching": [0.0, 0.0, 1.0]})
    extractor, _ = make_extractor(CHUNKS, vectors)

    features = extractor.update("thanks for watching", current_index=2, window_time=1.0, asr_confidence=0.9)

    assert features.semantic_next == features.semantic_current
    assert features.semantic_delta == 0.0


def test_empty_transcript_short_circuits_semantic_scoring_without_encoding(make_extractor):
    extractor, stub = make_extractor(CHUNKS, dict(CHUNK_VECTORS))

    features = extractor.update("", current_index=0, window_time=1.0, asr_confidence=0.0)

    assert features.semantic_current == 0.0
    assert features.semantic_next == 0.0
    assert stub.encode_calls == []


def test_time_in_chunk_resets_when_current_index_changes(make_extractor):
    vectors = dict(CHUNK_VECTORS, **{"welcome everyone": [1.0, 0.0, 0.0], "today we explore": [0.0, 1.0, 0.0]})
    extractor, _ = make_extractor(CHUNKS, vectors, history_windows=1)

    extractor.update("welcome everyone", current_index=0, window_time=10.0, asr_confidence=0.9)
    same_chunk = extractor.update("welcome everyone", current_index=0, window_time=12.0, asr_confidence=0.9)
    new_chunk = extractor.update("today we explore", current_index=1, window_time=13.0, asr_confidence=0.9)

    assert same_chunk.time_in_chunk == pytest.approx(2.0)
    assert new_chunk.time_in_chunk == pytest.approx(0.0)


def test_tokens_seen_ratio_accumulates_within_a_chunk_and_resets_on_change(make_extractor):
    vectors = dict(CHUNK_VECTORS, **{"welcome": [1.0, 0.0, 0.0], "everyone": [1.0, 0.0, 0.0], "today": [0.0, 1.0, 0.0]})
    extractor, _ = make_extractor(CHUNKS, vectors, history_windows=1)

    first = extractor.update("welcome", current_index=0, window_time=1.0, asr_confidence=0.9)
    second = extractor.update("everyone", current_index=0, window_time=2.0, asr_confidence=0.9)
    after_advance = extractor.update("today", current_index=1, window_time=3.0, asr_confidence=0.9)

    assert first.tokens_seen_ratio == pytest.approx(0.5)
    assert second.tokens_seen_ratio == pytest.approx(1.0)
    assert after_advance.tokens_seen_ratio == pytest.approx(1 / 3)


def test_lexical_rolling_mean_covers_last_three_windows_and_does_not_reset_on_advance(make_extractor):
    vectors = dict(
        CHUNK_VECTORS,
        **{"xyz": [1.0, 0.0, 0.0], "welcome everyone": [1.0, 0.0, 0.0], "today we explore": [0.0, 1.0, 0.0]},
    )
    extractor, _ = make_extractor(CHUNKS, vectors, history_windows=1)

    scores = []
    scores.append(extractor.update("xyz", current_index=0, window_time=1.0, asr_confidence=0.9).lexical_current)
    scores.append(
        extractor.update("welcome everyone", current_index=0, window_time=2.0, asr_confidence=0.9).lexical_current
    )
    features = extractor.update("today we explore", current_index=1, window_time=3.0, asr_confidence=0.9)
    scores.append(features.lexical_current)

    assert features.lexical_rolling_mean == pytest.approx(mean_of(scores))


def mean_of(values):
    return sum(values) / len(values)


def test_rejects_empty_chunks(make_extractor):
    with pytest.raises(ValueError, match="at least one"):
        make_extractor([], {})[0]


def test_rejects_non_positive_history_windows(make_extractor):
    with pytest.raises(ValueError, match="history_windows"):
        make_extractor(["welcome"], {"welcome": [1.0]}, history_windows=0)


def test_rejects_non_positive_rolling_window(make_extractor):
    with pytest.raises(ValueError, match="rolling_window"):
        make_extractor(["welcome"], {"welcome": [1.0]}, rolling_window=0)


def test_reset_clears_state_for_reuse_across_recordings(make_extractor):
    extractor, _ = make_extractor(CHUNKS, dict(CHUNK_VECTORS))
    extractor.update("welcome everyone", current_index=0, window_time=10.0, asr_confidence=0.9)
    extractor.update("today we explore", current_index=1, window_time=20.0, asr_confidence=0.9)

    extractor.reset()
    features = extractor.update("welcome everyone", current_index=0, window_time=0.5, asr_confidence=0.9)

    assert features.time_in_chunk == pytest.approx(0.0)
    assert features.tokens_seen_ratio == pytest.approx(1.0)
