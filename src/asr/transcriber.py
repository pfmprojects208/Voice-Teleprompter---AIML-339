import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


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
                    segments, _ = self.model.transcribe(window, language="en",
                                                        beam_size=1, vad_filter=False)
                    text = " ".join(s.text.strip() for s in segments)
                    if text:
                        yield text

    def stop(self):
        self._stop_event.set()
