import math
import wave
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.asr.transcriber import Transcriber


class _FakeSegment:
    def __init__(self, text, avg_logprob):
        self.text = text
        self.avg_logprob = avg_logprob


class _StubWhisperModel:
    """Deterministic stand-in for WhisperModel: one canned response per call, in order."""

    def __init__(self, responses):
        self._responses = responses
        self.calls = 0

    def transcribe(self, window, language="en", beam_size=1, vad_filter=False):
        segments = self._responses[self.calls]
        self.calls += 1
        return segments, None


def _write_wav(path, num_samples, sample_rate=16000, sampwidth=2):
    audio = np.zeros(num_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


@pytest.fixture
def make_transcriber(monkeypatch):
    def _make(responses, **kwargs):
        stub = _StubWhisperModel(responses)
        monkeypatch.setattr("src.asr.transcriber.WhisperModel", lambda *a, **kw: stub)
        return Transcriber(**kwargs), stub

    return _make


def test_stream_file_yields_one_window_transcript_per_full_second(tmp_path, make_transcriber):
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, num_samples=16000 * 2)  # exactly 2 windows

    responses = [[_FakeSegment("hello", -0.1)], [_FakeSegment("world", -0.3)]]
    transcriber, _ = make_transcriber(responses)

    results = list(transcriber.stream_file(str(wav_path)))

    assert [r.text for r in results] == ["hello", "world"]
    assert results[0].window_time == pytest.approx(1.0)
    assert results[1].window_time == pytest.approx(2.0)


def test_stream_file_drops_trailing_partial_window(tmp_path, make_transcriber):
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, num_samples=16000 + 8000)  # 1.5s -> only 1 full window

    transcriber, stub = make_transcriber([[_FakeSegment("hello", -0.1)]])

    results = list(transcriber.stream_file(str(wav_path)))

    assert len(results) == 1
    assert stub.calls == 1


def test_stream_file_skips_windows_with_empty_transcript(tmp_path, make_transcriber):
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, num_samples=16000 * 2)

    responses = [[], [_FakeSegment("world", -0.2)]]
    transcriber, _ = make_transcriber(responses)

    results = list(transcriber.stream_file(str(wav_path)))

    assert [r.text for r in results] == ["world"]
    assert results[0].window_time == pytest.approx(2.0)


def test_stream_file_confidence_is_exp_of_mean_avg_logprob(tmp_path, make_transcriber):
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, num_samples=16000)

    responses = [[_FakeSegment("hi", -0.2), _FakeSegment("there", -0.4)]]
    transcriber, _ = make_transcriber(responses)

    [result] = list(transcriber.stream_file(str(wav_path)))

    assert result.confidence == pytest.approx(math.exp(-0.3))


def test_stream_file_rejects_mismatched_sample_rate(tmp_path, make_transcriber):
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, num_samples=8000, sample_rate=8000)

    transcriber, _ = make_transcriber([[]])

    with pytest.raises(ValueError, match="8000Hz"):
        list(transcriber.stream_file(str(wav_path)))


def test_stream_file_rejects_non_16bit_audio(tmp_path, make_transcriber):
    wav_path = tmp_path / "audio.wav"
    _write_wav(wav_path, num_samples=16000, sampwidth=1)

    transcriber, _ = make_transcriber([[]])

    with pytest.raises(ValueError, match="16-bit"):
        list(transcriber.stream_file(str(wav_path)))
