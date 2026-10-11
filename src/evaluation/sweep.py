"""Parameter sweep for the two issues flagged in the report:

1. B1/B2's `history_windows=4` is too short for the dataset's actual
   chunk length (~5-6s), so the lexical score rarely reaches threshold.
2. B3's `confirm_windows` hysteresis was added but never tuned/evaluated
   in the final run.

ASR transcription (the slow part) is cached to disk per recording and
reused across every parameter value, so re-running this sweep is cheap.

Usage (from src/evaluation/):
    python sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.chunker import load_chunks
from alignment.matcher import BaselineMatcher, ClassifierMatcher, HybridMatcher
from asr.transcriber import Transcriber, WindowTranscript
from evaluation.ground_truth import GroundTruthPointer
from evaluation.evaluate import (
    CLASSIFIER_MODEL_PATH, CONDITIONS, LABELS_DIR, RAW_DIR, SCRIPT_NUMBERS,
    SCRIPTS_DIR, Bucket, compute_metrics, replay,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "evaluation" / "_transcript_cache"
OUT_CSV = REPO_ROOT / "data" / "evaluation" / "sweep_results.csv"

HISTORY_WINDOWS_VALUES = [4, 5, 6, 7, 8]
CONFIRM_WINDOWS_VALUES = [1, 2, 3, 4, 5]


def load_recordings(speaker: str = "A"):
    """Transcribes (or loads from cache) every recording once; returns
    {(script_num, condition): (chunks, windows, transitions)}."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    recordings = {}
    transcriber = None
    for script_num in SCRIPT_NUMBERS:
        chunks = load_chunks(SCRIPTS_DIR / f"script{script_num}.txt")
        for condition in CONDITIONS:
            audio_path = RAW_DIR / f"script{script_num}_speaker{speaker}_{condition}.wav"
            labels_path = LABELS_DIR / f"script{script_num}_speaker{speaker}_{condition}.json"
            if not audio_path.exists() or not labels_path.exists():
                continue

            cache_path = CACHE_DIR / f"script{script_num}_speaker{speaker}_{condition}.json"
            if cache_path.exists():
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                windows = [WindowTranscript(**w) for w in data]
            else:
                if transcriber is None:
                    transcriber = Transcriber()
                print(f"[script{script_num}/{condition}] transcribing (uncached)...")
                windows = list(transcriber.stream_file(str(audio_path)))
                cache_path.write_text(
                    json.dumps([w.__dict__ for w in windows]), encoding="utf-8"
                )

            transitions = json.loads(labels_path.read_text(encoding="utf-8"))["chunk_transitions"]
            recordings[(script_num, condition)] = (chunks, windows, transitions)
    return recordings


def evaluate_config(recordings, build_matcher):
    """`build_matcher(chunks)` must return a fresh matcher instance for
    one script's chunk list. Returns (overall_metrics, per_condition_metrics)."""
    buckets = {c: Bucket() for c in CONDITIONS}
    for (_script_num, condition), (chunks, windows, transitions) in recordings.items():
        matcher = build_matcher(chunks)
        pointer = GroundTruthPointer(transitions)
        buckets[condition].add_recording(replay(windows, matcher, pointer))

    present = [c for c in CONDITIONS if buckets[c].windows]
    overall = buckets[present[0]].merged_with([buckets[c] for c in present[1:]])
    return compute_metrics(overall), {c: compute_metrics(buckets[c]) for c in present}


def main():
    recordings = load_recordings()
    rows = []

    print("\n== B1 baseline: sweeping history_windows ==")
    for hw in HISTORY_WINDOWS_VALUES:
        overall, _ = evaluate_config(
            recordings, lambda chunks, hw=hw: BaselineMatcher(chunks, history_windows=hw)
        )
        print(f"  history_windows={hw}  acc={overall.accuracy:.3f}  far={overall.false_advance_rate:.3f}  "
              f"f1={overall.f1_advance:.3f}  detected={overall.n_latencies_measured}/{overall.n_transitions}")
        rows.append(dict(matcher="baseline", param="history_windows", value=hw,
                          accuracy=overall.accuracy, far=overall.false_advance_rate,
                          f1=overall.f1_advance, detected=overall.n_latencies_measured,
                          n_transitions=overall.n_transitions))

    print("\n== B2 hybrid: sweeping history_windows ==")
    for hw in HISTORY_WINDOWS_VALUES:
        overall, _ = evaluate_config(
            recordings, lambda chunks, hw=hw: HybridMatcher(chunks, history_windows=hw)
        )
        print(f"  history_windows={hw}  acc={overall.accuracy:.3f}  far={overall.false_advance_rate:.3f}  "
              f"f1={overall.f1_advance:.3f}  detected={overall.n_latencies_measured}/{overall.n_transitions}")
        rows.append(dict(matcher="hybrid", param="history_windows", value=hw,
                          accuracy=overall.accuracy, far=overall.false_advance_rate,
                          f1=overall.f1_advance, detected=overall.n_latencies_measured,
                          n_transitions=overall.n_transitions))

    if CLASSIFIER_MODEL_PATH.exists():
        print("\n== B3 classifier: sweeping confirm_windows ==")
        for cw in CONFIRM_WINDOWS_VALUES:
            overall, _ = evaluate_config(
                recordings,
                lambda chunks, cw=cw: ClassifierMatcher(
                    chunks, CLASSIFIER_MODEL_PATH, confirm_windows=cw
                ),
            )
            print(f"  confirm_windows={cw}  acc={overall.accuracy:.3f}  far={overall.false_advance_rate:.3f}  "
                  f"f1={overall.f1_advance:.3f}  detected={overall.n_latencies_measured}/{overall.n_transitions}")
            rows.append(dict(matcher="classifier", param="confirm_windows", value=cw,
                              accuracy=overall.accuracy, far=overall.false_advance_rate,
                              f1=overall.f1_advance, detected=overall.n_latencies_measured,
                              n_transitions=overall.n_transitions))

    import csv
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["matcher", "param", "value", "accuracy", "far", "f1", "detected", "n_transitions"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()
