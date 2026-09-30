"""Computes the 8 advance-classifier features described in the README.

Kept separate from `alignment.matcher` so the exact same feature
computation can be shared between the training-data extraction pipeline
(chunk pointer driven by ground-truth transition timestamps) and, later,
a learned B3 matcher at inference time (chunk pointer driven by its own
decisions) - avoiding train/serve skew.

`current_index` and `asr_confidence` are supplied by the caller on every
`update()` call: the chunk pointer isn't this class's job, and only the
ASR layer has access to the Whisper segment log-probabilities confidence
is derived from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

from rapidfuzz.fuzz import token_sort_ratio
from sentence_transformers import SentenceTransformer, util

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


@dataclass
class Features:
    lexical_current: float
    semantic_current: float
    semantic_next: float
    semantic_delta: float
    time_in_chunk: float
    tokens_seen_ratio: float
    lexical_rolling_mean: float
    asr_confidence: float


class FeatureExtractor:
    """Computes per-window features against a fixed list of script chunks.

    Two internal accumulators are reset whenever `current_index` changes
    (mirrors BaselineMatcher/HybridMatcher clearing their transcript
    history on advance):
      - the short transcript-history buffer used for lexical/semantic
        similarity against the current chunk.
      - the running "seen so far" transcript used for tokens_seen_ratio.

    The rolling mean of lexical scores (feature 7) is NOT reset on chunk
    change: it's a short-term trend signal, and a window or two spanning
    a just-happened transition is itself informative.
    """

    def __init__(
        self,
        chunks: list[str],
        model_name: str = "all-MiniLM-L6-v2",
        history_windows: int = 4,
        rolling_window: int = 3,
    ):
        if not chunks:
            raise ValueError("chunks must contain at least one script segment")
        if history_windows < 1:
            raise ValueError("history_windows must be at least 1")
        if rolling_window < 1:
            raise ValueError("rolling_window must be at least 1")

        self.chunks = chunks
        self.history_windows = history_windows
        self.rolling_window = rolling_window

        self._model = SentenceTransformer(model_name)
        self._chunk_embeddings = self._model.encode(chunks, convert_to_tensor=True)
        self._chunk_token_sets = [_tokenize(chunk) for chunk in chunks]

        self._current_index: int | None = None
        self._transcript_history: list[str] = []
        self._lexical_history: list[float] = []
        self._chunk_transcript = ""
        self._chunk_start_time: float | None = None

    def update(
        self, transcript: str, current_index: int, window_time: float, asr_confidence: float
    ) -> Features:
        """Feeds one ASR window and returns its feature vector.

        `window_time` is that window's position (seconds) into the
        recording - used as the elapsed-time anchor, and captured as the
        chunk's start time the first time `current_index` is seen.
        """
        if current_index != self._current_index:
            self._current_index = current_index
            self._transcript_history = []
            self._chunk_transcript = ""
            self._chunk_start_time = window_time

        self._transcript_history.append(transcript)
        self._transcript_history = self._transcript_history[-self.history_windows :]
        transcript_window = " ".join(self._transcript_history).strip()

        self._chunk_transcript = (self._chunk_transcript + " " + transcript).strip()

        lexical_current = token_sort_ratio(transcript_window, self.chunks[current_index])
        self._lexical_history.append(lexical_current)
        self._lexical_history = self._lexical_history[-self.rolling_window :]

        semantic_current, semantic_next = self._semantic_scores(transcript_window, current_index)

        return Features(
            lexical_current=lexical_current,
            semantic_current=semantic_current,
            semantic_next=semantic_next,
            semantic_delta=semantic_next - semantic_current,
            time_in_chunk=window_time - self._chunk_start_time,
            tokens_seen_ratio=self._tokens_seen_ratio(current_index),
            lexical_rolling_mean=mean(self._lexical_history),
            asr_confidence=asr_confidence,
        )

    def _semantic_scores(self, transcript_window: str, current_index: int) -> tuple[float, float]:
        if not transcript_window:
            return 0.0, 0.0

        transcript_embedding = self._model.encode(transcript_window, convert_to_tensor=True)
        semantic_current = util.cos_sim(
            transcript_embedding, self._chunk_embeddings[current_index]
        ).item()

        next_index = current_index + 1
        if next_index >= len(self.chunks):
            # No next chunk to advance into - nothing to signal readiness for.
            return semantic_current, semantic_current

        semantic_next = util.cos_sim(
            transcript_embedding, self._chunk_embeddings[next_index]
        ).item()
        return semantic_current, semantic_next

    def _tokens_seen_ratio(self, current_index: int) -> float:
        chunk_tokens = self._chunk_token_sets[current_index]
        if not chunk_tokens:
            return 0.0
        seen_tokens = _tokenize(self._chunk_transcript)
        return len(chunk_tokens & seen_tokens) / len(chunk_tokens)

    def reset(self):
        """Clears per-recording state so one instance can be reused across
        recordings that share the same chunk list (e.g. several speakers
        reading the same literal script), without reloading the model."""
        self._current_index = None
        self._transcript_history = []
        self._lexical_history = []
        self._chunk_transcript = ""
        self._chunk_start_time = None
