from rapidfuzz.fuzz import token_sort_ratio
from sentence_transformers import SentenceTransformer, util


class BaselineMatcher:
    """Advance through script chunks using a bounded ASR transcript history."""

    def __init__(
        self, chunks: list[str], threshold: float = 70.0, history_windows: int = 4
    ):
        if not chunks:
            raise ValueError("chunks must contain at least one script segment")
        if history_windows < 1:
            raise ValueError("history_windows must be at least 1")
        self.chunks = chunks
        self.threshold = threshold
        self.history_windows = history_windows
        self.current_index = 0
        self._finished = False
        self._transcript_history: list[str] = []

    @property
    def current_chunk(self) -> str:
        return self.chunks[self.current_index]

    @property
    def finished(self) -> bool:
        return self._finished

    def update(self, transcript: str) -> tuple[str, float, bool]:
        if self.finished:
            return "", 0.0, False

        self._transcript_history.append(transcript)
        self._transcript_history = self._transcript_history[-self.history_windows :]
        transcript_window = " ".join(self._transcript_history)
        score = token_sort_ratio(transcript_window, self.current_chunk)
        advanced = False

        if score >= self.threshold:
            if self.current_index < len(self.chunks) - 1:
                self.current_index += 1
                advanced = True
                self._transcript_history.clear()
            else:
                self._finished = True

        return self.current_chunk, score, advanced

    def reset(self):
        self.current_index = 0
        self._finished = False
        self._transcript_history.clear()


class HybridMatcher:
    """Advance through script chunks using lexical matching with a semantic fallback.

    Mirrors BaselineMatcher's lexical scoring, but when the lexical score drops
    below `secondary_threshold` (signalling paraphrase rather than a bad match),
    semantic cosine similarity decides whether to advance instead.
    """

    def __init__(
        self,
        chunks: list[str],
        threshold: float = 70.0,
        secondary_threshold: float = 40.0,
        semantic_threshold: float = 0.6,
        history_windows: int = 4,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        if not chunks:
            raise ValueError("chunks must contain at least one script segment")
        if history_windows < 1:
            raise ValueError("history_windows must be at least 1")
        if secondary_threshold >= threshold:
            raise ValueError("secondary_threshold must be lower than threshold")

        self.chunks = chunks
        self.threshold = threshold
        self.secondary_threshold = secondary_threshold
        self.semantic_threshold = semantic_threshold
        self.history_windows = history_windows
        self.current_index = 0
        self._finished = False
        self._transcript_history: list[str] = []

        self._model = SentenceTransformer(model_name)
        self._chunk_embeddings = self._model.encode(chunks, convert_to_tensor=True)

    @property
    def current_chunk(self) -> str:
        return self.chunks[self.current_index]

    @property
    def finished(self) -> bool:
        return self._finished

    def update(self, transcript: str) -> tuple[str, float, bool]:
        if self.finished:
            return "", 0.0, False

        self._transcript_history.append(transcript)
        self._transcript_history = self._transcript_history[-self.history_windows :]
        transcript_window = " ".join(self._transcript_history)

        lexical_score = token_sort_ratio(transcript_window, self.current_chunk)
        score = lexical_score
        should_advance = lexical_score >= self.threshold

        if not should_advance and lexical_score < self.secondary_threshold:
            semantic_score = self._semantic_similarity(transcript_window)
            score = semantic_score * 100
            should_advance = semantic_score >= self.semantic_threshold

        advanced = False
        if should_advance:
            if self.current_index < len(self.chunks) - 1:
                self.current_index += 1
                advanced = True
                self._transcript_history.clear()
            else:
                self._finished = True

        return self.current_chunk, score, advanced

    def _semantic_similarity(self, transcript_window: str) -> float:
        transcript_embedding = self._model.encode(transcript_window, convert_to_tensor=True)
        chunk_embedding = self._chunk_embeddings[self.current_index]
        return util.cos_sim(transcript_embedding, chunk_embedding).item()

    def reset(self):
        self.current_index = 0
        self._finished = False
        self._transcript_history.clear()
