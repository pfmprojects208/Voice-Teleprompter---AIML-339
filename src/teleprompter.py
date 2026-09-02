from alignment.chunker import load_chunks
from alignment.matcher import BaselineMatcher
from asr.transcriber import Transcriber
from display.server import DisplayServer


def run(script_path: str, threshold: float = 70.0):
    chunks = load_chunks(script_path)
    matcher = BaselineMatcher(chunks, threshold=threshold)
    transcriber = Transcriber()
    server = DisplayServer()
    server.start()

    print(f"Loaded {len(chunks)} chunks. Threshold: {threshold}")
    print("Open src/display/index.html in your browser, then start speaking.\n")

    try:
        for transcript in transcriber.stream():
            chunk, score, advanced = matcher.update(transcript)
            server.send(chunk, matcher.current_index, score)
            tag = " → ADVANCE" if advanced else ""
            print(f"[{score:5.1f}] {transcript:<60}{tag}")

            if matcher.finished:
                print("Script complete.")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    run("../scripts/sample_script.txt")
