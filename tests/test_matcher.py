from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.alignment.matcher import BaselineMatcher, ClassifierMatcher, HybridMatcher


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


class _StubFeatureExtractor:
    """Deterministic stand-in for evaluation.features.FeatureExtractor:
    returns a fixed feature vector regardless of transcript, and tracks
    calls so tests can assert on them."""

    def __init__(self, chunks, model_name="all-MiniLM-L6-v2", history_windows=4, rolling_window=3):
        self.chunks = chunks
        self.reset_calls = 0
        self.update_calls: list[tuple] = []

    def update(self, transcript, current_index, window_time, asr_confidence):
        self.update_calls.append((transcript, current_index, window_time, asr_confidence))
        return SimpleNamespace(
            lexical_current=1.0, semantic_current=2.0, semantic_next=3.0,
            semantic_delta=4.0, time_in_chunk=5.0, tokens_seen_ratio=6.0,
            lexical_rolling_mean=7.0, asr_confidence=asr_confidence,
        )

    def reset(self):
        self.reset_calls += 1


class _StubClassifierModel:
    """Deterministic stand-in for the trained sklearn pipeline: always
    predicts the given class and advance-class probability."""

    def __init__(self, predict_value: int, proba_advance: float):
        self.predict_value = predict_value
        self.proba_advance = proba_advance
        self.last_X: np.ndarray | None = None

    def predict(self, X):
        self.last_X = X
        return [self.predict_value]

    def predict_proba(self, X):
        self.last_X = X
        return np.array([[1 - self.proba_advance, self.proba_advance]])


CLASSIFIER_CHUNKS = ["welcome everyone", "today we explore"]


@pytest.fixture
def make_classifier_matcher(monkeypatch):
    def _make(chunks, predict_value, proba_advance, feature_names=("lexical_current", "asr_confidence")):
        model = _StubClassifierModel(predict_value, proba_advance)
        bundle = {"model": model, "features": list(feature_names)}
        monkeypatch.setattr("src.alignment.matcher.joblib.load", lambda path: bundle)

        extractors: list[_StubFeatureExtractor] = []

        def _factory(chunks_, **kwargs):
            extractor = _StubFeatureExtractor(chunks_, **kwargs)
            extractors.append(extractor)
            return extractor

        monkeypatch.setattr("src.alignment.matcher.FeatureExtractor", _factory)

        matcher = ClassifierMatcher(chunks, model_path="unused.joblib")
        return matcher, model, extractors[0]

    return _make


def test_classifier_advances_when_model_predicts_advance(make_classifier_matcher):
    matcher, _, _ = make_classifier_matcher(CLASSIFIER_CHUNKS, predict_value=1, proba_advance=0.9)

    chunk, score, advanced = matcher.update("some transcript", window_time=1.0, asr_confidence=0.8)

    assert advanced
    assert chunk == "today we explore"
    assert matcher.current_index == 1
    assert score == pytest.approx(90.0)
    assert not matcher.finished


def test_classifier_holds_when_model_predicts_hold(make_classifier_matcher):
    matcher, _, _ = make_classifier_matcher(CLASSIFIER_CHUNKS, predict_value=0, proba_advance=0.2)

    chunk, score, advanced = matcher.update("some transcript")

    assert not advanced
    assert chunk == "welcome everyone"
    assert matcher.current_index == 0
    assert score == pytest.approx(20.0)


def test_classifier_matching_final_chunk_marks_finished(make_classifier_matcher):
    matcher, _, _ = make_classifier_matcher(CLASSIFIER_CHUNKS, predict_value=1, proba_advance=0.9)
    matcher.update("first")

    chunk, score, advanced = matcher.update("second")

    assert not advanced
    assert matcher.finished
    assert chunk == "today we explore"
    assert matcher.current_index == 1


def test_classifier_feature_vector_uses_named_columns_in_order(make_classifier_matcher):
    matcher, model, _ = make_classifier_matcher(
        CLASSIFIER_CHUNKS, predict_value=0, proba_advance=0.0,
        feature_names=("asr_confidence", "lexical_current"),
    )

    matcher.update("hello", window_time=2.0, asr_confidence=0.42)

    assert model.last_X.tolist() == [[0.42, 1.0]]


def test_classifier_reset_resets_state_and_feature_extractor(make_classifier_matcher):
    matcher, _, extractor = make_classifier_matcher(CLASSIFIER_CHUNKS, predict_value=1, proba_advance=0.9)
    matcher.update("first")

    matcher.reset()

    assert matcher.current_index == 0
    assert not matcher.finished
    assert extractor.reset_calls == 1


def test_classifier_finished_matcher_returns_empty_without_calling_extractor(make_classifier_matcher):
    matcher, _, extractor = make_classifier_matcher(CLASSIFIER_CHUNKS, predict_value=1, proba_advance=0.9)
    matcher.update("first")
    matcher.update("second")
    assert matcher.finished
    calls_before = len(extractor.update_calls)

    chunk, score, advanced = matcher.update("third")

    assert chunk == ""
    assert score == 0.0
    assert not advanced
    assert len(extractor.update_calls) == calls_before


def test_classifier_empty_script_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        ClassifierMatcher([], model_path="unused.joblib")
