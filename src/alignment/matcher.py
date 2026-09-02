from rapidfuzz.fuzz import token_sort_ratio


class BaselineMatcher:

    def __init__(self, chunks: list[str], threshold: float = 70.0):
        self.chunks = chunks
        self.threshold = threshold
        self.current_index = 0

    @property
    def current_chunk(self) -> str:
        return self.chunks[self.current_index]

    @property
    def finished(self) -> bool:
        return self.current_index >= len(self.chunks)

    def update(self, transcript: str) -> tuple[str, float, bool]:
        if self.finished:
            return "", 0.0, False

        score = token_sort_ratio(transcript, self.current_chunk)
        advanced = False

        if score >= self.threshold and self.current_index < len(self.chunks) - 1:
            self.current_index += 1
            advanced = True

        return self.current_chunk, score, advanced

    def reset(self):
        self.current_index = 0
