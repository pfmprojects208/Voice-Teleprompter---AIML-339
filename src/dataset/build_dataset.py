"""Builds the B3 training dataset: one row per ASR window, per recording.

Replays each recording through the same windowed pipeline the live system
uses (`Transcriber.stream_file`), computes the 8 classifier features for
every window (`FeatureExtractor`), and labels it `advance` using the
verified ground truth (`GroundTruthPointer` over data/labels/*.json).

Usage (from src/dataset/):
    python build_dataset.py --speaker A
    python build_dataset.py --speaker A --script 1 --condition literal

Writes data/features/speaker<ID>.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asr.transcriber import Transcriber
from evaluation.features import FeatureExtractor
from evaluation.ground_truth import GroundTruthPointer

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "data" / "scripts"
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"
FEATURES_DIR = REPO_ROOT / "data" / "features"

SCRIPT_NUMBERS = ["1", "2", "3"]
CONDITIONS = ["literal", "paraphrase", "skip"]

FIELDNAMES = [
    "recording", "script", "speaker", "condition", "chunk_index", "window_time",
    "lexical_current", "semantic_current", "semantic_next", "semantic_delta",
    "time_in_chunk", "tokens_seen_ratio", "lexical_rolling_mean", "asr_confidence",
    "advance",
]


def load_chunks(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_recording_rows(
    transcriber: Transcriber,
    extractor: FeatureExtractor,
    script_num: str,
    speaker: str,
    condition: str,
) -> list[dict]:
    """Turns one recording into its feature rows. The script's ORIGINAL
    chunk list drives `extractor` regardless of condition, since that's
    what the teleprompter always displays - chunk_transitions in the
    label file are keyed to those same original indices."""
    audio_path = RAW_DIR / f"script{script_num}_speaker{speaker}_{condition}.wav"
    labels_path = LABELS_DIR / f"script{script_num}_speaker{speaker}_{condition}.json"
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)

    transitions = json.loads(labels_path.read_text(encoding="utf-8"))["chunk_transitions"]
    pointer = GroundTruthPointer(transitions)

    rows = []
    for window in transcriber.stream_file(str(audio_path)):
        current_index, advance = pointer.update(window.window_time)
        features = extractor.update(
            window.text, current_index, window.window_time, window.confidence
        )
        rows.append({
            "recording": audio_path.name,
            "script": script_num,
            "speaker": speaker,
            "condition": condition,
            "chunk_index": current_index,
            "window_time": window.window_time,
            "advance": int(advance),
            **asdict(features),
        })
    return rows


def build_dataset(
    speaker: str, scripts: list[str] = SCRIPT_NUMBERS, conditions: list[str] = CONDITIONS
) -> list[dict]:
    transcriber = Transcriber()
    all_rows: list[dict] = []

    for script_num in scripts:
        chunks = load_chunks(SCRIPTS_DIR / f"script{script_num}.txt")
        extractor = FeatureExtractor(chunks)

        for condition in conditions:
            print(f"[script{script_num} / {condition}]")
            try:
                rows = build_recording_rows(transcriber, extractor, script_num, speaker, condition)
            except FileNotFoundError as e:
                print(f"  skipped, missing file: {e}")
                continue
            print(f"  {len(rows)} windows, {sum(r['advance'] for r in rows)} advances")
            all_rows.extend(rows)
            extractor.reset()

    return all_rows


def write_csv(rows: list[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", required=True)
    parser.add_argument("--script", choices=SCRIPT_NUMBERS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    scripts = [args.script] if args.script else SCRIPT_NUMBERS
    conditions = [args.condition] if args.condition else CONDITIONS

    rows = build_dataset(args.speaker, scripts, conditions)
    out_path = args.out or FEATURES_DIR / f"speaker{args.speaker}.csv"
    write_csv(rows, out_path)
    print(f"\n{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
