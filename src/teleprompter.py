import json

from alignment.chunker import load_chunks
from alignment.matcher import BaselineMatcher, HybridMatcher
from asr.transcriber import Transcriber
from display.server import DisplayServer


class MatcherRegistry:
    """Holds every available matcher and tracks which one is currently active."""

    def __init__(self, matchers: dict, default: str):
        self._matchers = matchers
        self.active_name = default

    @property
    def active(self):
        return self._matchers[self.active_name]

    def select(self, name: str) -> bool:
        if name not in self._matchers or name == self.active_name:
            return False
        self._matchers[name].reset()
        self.active_name = name
        return True


def run(script_path: str, threshold: float = 70.0):
    chunks = load_chunks(script_path)
    matchers = {
        "baseline": BaselineMatcher(chunks, threshold=threshold),
        "hybrid": HybridMatcher(chunks, threshold=threshold),
    }
    registry = MatcherRegistry(matchers, default="baseline")

    def handle_message(raw_message: str):
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        if data.get("type") == "select_model" and registry.select(data.get("model", "")):
            print(f"\n→ Switched to model: {registry.active_name}\n")

    transcriber = Transcriber()
    server = DisplayServer(on_message=handle_message)
    server.start()

    print(f"Loaded {len(chunks)} chunks. Threshold: {threshold}")
    print("Open src/display/index.html in your browser, then start speaking.\n")

    try:
        for transcript in transcriber.stream():
            matcher = registry.active
            chunk, score, advanced = matcher.update(transcript)
            server.send(chunk, matcher.current_index, score, registry.active_name)
            tag = " → ADVANCE" if advanced else ""
            print(f"[{registry.active_name:8}][{score:5.1f}] {transcript:<60}{tag}")

            if matcher.finished:
                print("Script complete.")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    run("../scripts/sample_script.txt")
