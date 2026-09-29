import queue
import threading
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd


class Recorder:
    """Records mono 16-bit PCM audio from the microphone until stopped."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()

    def _callback(self, indata, frames, time, status):
        self._audio_queue.put(indata[:, 0].copy())

    def record_until_stopped(self) -> np.ndarray:
        """Blocks until `stop()` is called from another thread, then returns the audio."""
        self._stop_event.clear()
        chunks: list[np.ndarray] = []

        with sd.InputStream(samplerate=self.sample_rate, channels=1,
                             dtype="int16", callback=self._callback):
            while not self._stop_event.is_set():
                try:
                    chunks.append(self._audio_queue.get(timeout=0.5))
                except queue.Empty:
                    continue

        return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def save_wav(audio: np.ndarray, path: Path, sample_rate: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())
