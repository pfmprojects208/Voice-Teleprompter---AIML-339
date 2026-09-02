from alignment.chunker import load_chunks
from alignment.matcher import BaselineMatcher
from asr.transcriber import Transcriber


def run(script_path: str, threshold: float = 70.0):
    chunks = load_chunks(script_path)
    matcher = BaselineMatcher(chunks, threshold=threshold)
    transcriber = Transcriber()

    print(f"Loaded {len(chunks)} chunks. Threshold: {threshold}")
    print("Start speaking...\n")

    try:
        for transcript in transcriber.stream():
            chunk, score, advanced = matcher.update(transcript)
            tag = " → ADVANCE" if advanced else ""
            print(f"[{score:5.1f}] {transcript:<60}{tag}")
            print(f"       >>> {chunk}\n")

            if matcher.finished:
                print("Script complete.")
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        transcriber.stop()


if __name__ == "__main__":
    run("../scripts/sample_script.txt")
