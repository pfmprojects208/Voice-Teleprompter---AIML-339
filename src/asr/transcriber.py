from __future__ import annotations

import math
import queue
import threading
import wave
from dataclasses import dataclass
from statistics import mean

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


@dataclass
class WindowTranscript:
    text: str
    window_time: float
    confidence: float


class Transcriber:

    def __init__(self, model_size: str = "base", device: str = "cpu",
                 sample_rate: int = 16000, window_seconds: float = 1.0):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.sample_rate = sample_rate
        self.window_size = int(sample_rate * window_seconds)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()

    def _audio_callback(self, indata, frames, time, status):
        self._audio_queue.put(indata[:, 0].copy())

    def _transcribe_window(self, window: np.ndarray) -> tuple[str, float]:
        segments, _ = self.model.transcribe(window, language="en",
                                             beam_size=1, vad_filter=False)
        segments = list(segments)
        text = " ".join(s.text.strip() for s in segments).strip()
        confidence = math.exp(mean(s.avg_logprob for s in segments)) if segments else 0.0
        return text, confidence

    def stream(self):
        buffer = np.array([], dtype=np.float32)

        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                            dtype="float32", callback=self._audio_callback):
            while not self._stop_event.is_set():
                try:
                    chunk = self._audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                buffer = np.concatenate([buffer, chunk])

                if len(buffer) >= self.window_size:
                    window, buffer = buffer[:self.window_size], buffer[self.window_size:]
                    text, _ = self._transcribe_window(window)
                    if text:
                        yield text

    def stream_file(self, audio_path: str):
        """Replays a recorded .wav file through the same window-by-window
        pipeline as `stream()`, so a finished recording yields the exact
        transcript-per-window the live system would have produced -
        used to build the training dataset from data/raw/*.wav.

        Trailing audio shorter than one window is dropped, mirroring
        `stream()` (which never flushes a partial buffer either), and
        windows with empty transcript are skipped, since those never
        reach the matcher/feature extractor live.
        """
        with wave.open(audio_path, "rb") as wf:
            if wf.getframerate() != self.sample_rate:
                raise ValueError(
                    f"{audio_path} is {wf.getframerate()}Hz, expected {self.sample_rate}Hz"
                )
            if wf.getsampwidth() != 2:
                raise ValueError(f"{audio_path} is not 16-bit PCM")
            raw = wf.readframes(wf.getnframes())

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        samples_consumed = 0
        for start in range(0, len(audio) - self.window_size + 1, self.window_size):
            window = audio[start: start + self.window_size]
            samples_consumed += len(window)
            text, confidence = self._transcribe_window(window)
            if text:
                yield WindowTranscript(
                    text=text,
                    window_time=samples_consumed / self.sample_rate,
                    confidence=confidence,
                )

    def stop(self):
        self._stop_event.set()
