"""Forced alignment: turn one raw recording into ground-truth chunk transition times.

Usage (from src/dataset/):
    python run_alignment.py --script 1 --speaker A --condition literal
    python run_alignment.py --all --speaker A          # all 3 scripts x 3 conditions

Writes data/labels/script<N>_speaker<ID>_<condition>.json with, for every chunk
index of the ORIGINAL script, the timestamp (seconds) its reading began -
or null if that chunk was skipped in this recording.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from align import align_chunk_starts, transcribe_words

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "data" / "scripts"
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"

CONDITION_SUFFIX = {"literal": "", "paraphrase": "_paraphrase", "skip": "_skip"}


def load_chunks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def align_recording(script_num: str, speaker: str, condition: str, model_size: str = "base") -> Path:
    suffix = CONDITION_SUFFIX[condition]
    text_path = SCRIPTS_DIR / f"script{script_num}{suffix}.txt"
    audio_path = RAW_DIR / f"script{script_num}_speaker{speaker}_{condition}.wav"
    plan_path = SCRIPTS_DIR / f"script{script_num}_plan.json"

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    chunks = load_chunks(text_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    original_total = plan["total_chunks"]
    skip_chunks = sorted(plan["skip_chunks"]) if condition == "skip" else []

    print(f"  transcribing {audio_path.name} ...")
    hyp_words = transcribe_words(str(audio_path), model_size)
    starts = align_chunk_starts(chunks, hyp_words)

    transitions: dict[str, float | None] = {}
    if condition == "skip":
        skip_start, skip_len = skip_chunks[0], len(skip_chunks)
        for local_idx, ts in enumerate(starts):
            orig_idx = local_idx if local_idx < skip_start else local_idx + skip_len
            transitions[str(orig_idx)] = ts
        for i in skip_chunks:
            transitions[str(i)] = None
    else:
        for local_idx, ts in enumerate(starts):
            transitions[str(local_idx)] = ts

    out = {
        "recording": audio_path.name,
        "condition": condition,
        "total_chunks_original": original_total,
        "skipped_chunks": skip_chunks,
        "chunk_transitions": {str(i): transitions.get(str(i)) for i in range(original_total)},
    }

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LABELS_DIR / f"script{script_num}_speaker{speaker}_{condition}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--script", choices=["1", "2", "3"])
    parser.add_argument("--condition", choices=list(CONDITION_SUFFIX))
    parser.add_argument("--all", action="store_true", help="align all 3 scripts x 3 conditions")
    args = parser.parse_args()

    jobs = []
    if args.all:
        for s in ["1", "2", "3"]:
            for c in CONDITION_SUFFIX:
                jobs.append((s, c))
    else:
        if not args.script or not args.condition:
            sys.exit("Provide --script and --condition, or use --all")
        jobs.append((args.script, args.condition))

    for script_num, condition in jobs:
        print(f"[script{script_num} / {condition}]")
        try:
            out_path = align_recording(script_num, args.speaker, condition)
            print(f"  -> {out_path}")
        except FileNotFoundError as e:
            print(f"  skipped, missing file: {e}")


if __name__ == "__main__":
    main()
