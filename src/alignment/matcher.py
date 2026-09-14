from rapidfuzz.fuzz import token_sort_ratio


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
