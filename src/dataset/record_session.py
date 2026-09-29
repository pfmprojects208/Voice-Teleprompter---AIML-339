"""Record one dataset take (one script, one speaker, one condition).

Usage (from src/dataset/):
    python record_session.py --script 1 --speaker A --condition literal
    python record_session.py --script 2 --speaker B --condition paraphrase
    python record_session.py --script 3 --speaker A --condition skip

Saves to data/raw/script<N>_speaker<ID>_<condition>.wav (mono, 16kHz, 16-bit PCM).
"""
import argparse
import sys
import threading
from pathlib import Path

from recorder import Recorder

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "data" / "scripts"
RAW_DIR = REPO_ROOT / "data" / "raw"

CONDITION_SUFFIX = {
    "literal": "",
    "paraphrase": "_paraphrase",
    "skip": "_skip",
}


def main():
    parser = argparse.ArgumentParser(description="Record one dataset take.")
    parser.add_argument("--script", required=True, choices=["1", "2", "3"])
    parser.add_argument("--speaker", required=True, help="Speaker id, e.g. A or B")
    parser.add_argument("--condition", required=True, choices=list(CONDITION_SUFFIX))
    args = parser.parse_args()

    suffix = CONDITION_SUFFIX[args.condition]
    text_path = SCRIPTS_DIR / f"script{args.script}{suffix}.txt"
    if not text_path.exists():
        sys.exit(f"Missing script file: {text_path}")

    out_path = RAW_DIR / f"script{args.script}_speaker{args.speaker}_{args.condition}.wav"
    if out_path.exists():
        confirm = input(f"{out_path.name} already exists. Overwrite? [y/N] ")
        if confirm.strip().lower() != "y":
            sys.exit("Aborted.")

    print(f"\nText to read: {text_path}")
    print(f"Output file:  {out_path}")
    print("\nOpen the text file above now and get ready to read it aloud.")
    input("Press Enter to START recording...")

    recorder = Recorder()
    result: dict = {}

    def _record():
        result["audio"] = recorder.record_until_stopped()

    thread = threading.Thread(target=_record)
    thread.start()
    input("\nRecording... press Enter to STOP.\n")
    recorder.stop()
    thread.join()

    audio = result["audio"]
    duration = len(audio) / recorder.sample_rate
    if duration < 1.0:
        sys.exit(f"Only captured {duration:.1f}s of audio — discarding, nothing saved.")

    Recorder.save_wav(audio, out_path, recorder.sample_rate)
    print(f"Saved {duration:.1f}s of audio to {out_path}")


if __name__ == "__main__":
    main()
