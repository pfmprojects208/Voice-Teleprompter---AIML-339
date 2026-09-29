from __future__ import annotations

import difflib
import re

from faster_whisper import WhisperModel

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def transcribe_words(audio_path: str, model_size: str = "base") -> list[tuple[str, float, float]]:
    """Transcribes an audio file and returns (word, start_time, end_time) for every word."""
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en", word_timestamps=True, beam_size=5)

    words = []
    for segment in segments:
        for word in segment.words:
            words.append((word.word.strip(), word.start, word.end))
    return words


def align_chunk_starts(
    chunks: list[str], hyp_words: list[tuple[str, float, float]]
) -> list[float | None]:
    """Aligns script chunks against transcribed words to find when each chunk started.

    Uses word-level sequence matching (difflib) between the reference script text
    and the ASR hypothesis, tolerant of ASR mistakes (insertions/deletions/substitutions).
    Returns one timestamp per chunk, or None if no matching word could be found.
    """
    ref_words: list[str] = []
    chunk_boundaries = [0]
    for chunk in chunks:
        ref_words.extend(_tokenize(chunk))
        chunk_boundaries.append(len(ref_words))

    hyp_tokens = []
    for word, _, _ in hyp_words:
        tokens = _tokenize(word)
        hyp_tokens.append(tokens[0] if tokens else "")

    matcher = difflib.SequenceMatcher(a=ref_words, b=hyp_tokens, autojunk=False)
    ref_to_hyp: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            ref_to_hyp[block.a + k] = block.b + k

    starts: list[float | None] = []
    for chunk_idx in range(len(chunks)):
        ref_start = chunk_boundaries[chunk_idx]
        ref_end = chunk_boundaries[chunk_idx + 1]

        hyp_idx = next((ref_to_hyp[r] for r in range(ref_start, ref_end) if r in ref_to_hyp), None)
        if hyp_idx is None:
            # chunk had no directly matched word (e.g. fully mis-transcribed) -
            # fall back to the next chunk's matched start, searching forward
            hyp_idx = next((ref_to_hyp[r] for r in range(ref_end, len(ref_words)) if r in ref_to_hyp), None)

        starts.append(hyp_words[hyp_idx][1] if hyp_idx is not None else None)

    return starts
