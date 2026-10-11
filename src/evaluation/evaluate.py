"""Evaluates B1, B2 and B3 against the labelled recordings.

Replays each `data/raw/*.wav` recording once through every available
matcher - `BaselineMatcher` (B1), `HybridMatcher` (B2), and
`ClassifierMatcher` (B3) when a trained model exists - scoring each
window against `data/labels/*.json` ground truth, and reports the
metrics from the README's "Evaluation Metrics" section:

  - chunk-position accuracy
  - false-advance rate (the asymmetric-cost error)
  - F1 on the advance class
  - mean advance latency
  - the same false-advance rate broken out per condition (literal /
    paraphrase / skip) - the paraphrase-tolerance trade-off curve

Latency is computed per recording (a transition queue that doesn't
survive past the end of its own recording) and only the resulting
latency VALUES are pooled across recordings/conditions - pooling raw
per-window results first and computing latency over the concatenation
would let a transition queue built against one recording's clock get
checked against a different recording's chunk positions.

Usage (from src/evaluation/):
    python evaluate.py
    python evaluate.py --speaker A --out ../../data/evaluation/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.chunker import load_chunks
from alignment.matcher import BaselineMatcher, ClassifierMatcher, HybridMatcher
from asr.transcriber import Transcriber, WindowTranscript
from evaluation.ground_truth import GroundTruthPointer
from evaluation.train_classifier import false_advance_rate

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "data" / "scripts"
RAW_DIR = REPO_ROOT / "data" / "raw"
LABELS_DIR = REPO_ROOT / "data" / "labels"
CLASSIFIER_MODEL_PATH = REPO_ROOT / "data" / "models" / "b3_classifier.joblib"

SCRIPT_NUMBERS = ["1", "2", "3"]
CONDITIONS = ["literal", "paraphrase", "skip"]


@dataclass
class WindowResult:
    window_time: float
    gt_index: int
    gt_advance: bool
    pred_index: int
    pred_advance: bool


def replay(windows: list[WindowTranscript], matcher, pointer: GroundTruthPointer) -> list[WindowResult]:
    """Feeds one recording's windows through `matcher`, scoring each
    against `pointer`. Mirrors `dataset/build_dataset.py`'s use of the
    same pointer, but also keeps the matcher's own displayed index to
    score position accuracy and latency, not just the advance label."""
    results = []
    for window in windows:
        _pre_gt_index, gt_advance = pointer.update(window.window_time)
        _chunk, _score, pred_advance = matcher.update(
            window.text, window.window_time, window.confidence
        )
        results.append(WindowResult(
            window_time=window.window_time,
            gt_index=pointer.current_index,
            gt_advance=gt_advance,
            pred_index=matcher.current_index,
            pred_advance=pred_advance,
        ))
    return results


def compute_latencies(results: list[WindowResult]) -> list[float]:
    """Seconds between each ground-truth transition and the first window
    where the matcher's displayed index has caught up to it, for ONE
    recording's results (`results` must come from a single `replay()`
    call - see module docstring on why pooling before this breaks it).
    A transition the matcher never reaches within the recording
    contributes no value - reported separately as a miss count, not
    averaged in as if instant."""
    pending: list[tuple[float, int]] = []
    latencies = []
    for r in results:
        if r.gt_advance:
            pending.append((r.window_time, r.gt_index))
        while pending and r.pred_index >= pending[0][1]:
            transition_time, _target_index = pending.pop(0)
            latencies.append(r.window_time - transition_time)
    return latencies


@dataclass
class Bucket:
    """Accumulates results across one or more recordings for a given
    (matcher, condition). `windows` is a flat pool (safe - each entry is
    already self-contained) but `latencies` must be computed per
    recording and only the resulting values pooled, so `add_recording`
    is the only way in."""
    windows: list[WindowResult] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)

    def add_recording(self, results: list[WindowResult]):
        self.windows.extend(results)
        self.latencies.extend(compute_latencies(results))

    def merged_with(self, others: list["Bucket"]) -> "Bucket":
        merged = Bucket(list(self.windows), list(self.latencies))
        for other in others:
            merged.windows.extend(other.windows)
            merged.latencies.extend(other.latencies)
        return merged


@dataclass
class Metrics:
    n_windows: int
    accuracy: float
    false_advance_rate: float
    f1_advance: float
    n_transitions: int
    n_latencies_measured: int
    mean_advance_latency: float | None


def compute_metrics(bucket: Bucket) -> Metrics:
    y_true = np.array([int(r.gt_advance) for r in bucket.windows])
    y_pred = np.array([int(r.pred_advance) for r in bucket.windows])
    accuracy = float(np.mean([r.pred_index == r.gt_index for r in bucket.windows]))

    return Metrics(
        n_windows=len(bucket.windows),
        accuracy=accuracy,
        false_advance_rate=false_advance_rate(y_true, y_pred),
        f1_advance=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        n_transitions=int(y_true.sum()),
        n_latencies_measured=len(bucket.latencies),
        mean_advance_latency=float(np.mean(bucket.latencies)) if bucket.latencies else None,
    )


def build_matchers(chunks: list[str], threshold: float, confirm_windows: int) -> dict:
    matchers = {
        "baseline": BaselineMatcher(chunks, threshold=threshold),
        "hybrid": HybridMatcher(chunks, threshold=threshold),
    }
    if CLASSIFIER_MODEL_PATH.exists():
        matchers["classifier"] = ClassifierMatcher(
            chunks, CLASSIFIER_MODEL_PATH, confirm_windows=confirm_windows
        )
    return matchers


def run_evaluation(speaker: str, threshold: float, confirm_windows: int = 1) -> dict[str, dict[str, Bucket]]:
    """Returns `results[matcher_name][condition]`, pooling recordings
    across all three scripts for that condition."""
    transcriber = Transcriber()
    matcher_names = ["baseline", "hybrid"] + (["classifier"] if CLASSIFIER_MODEL_PATH.exists() else [])
    results: dict[str, dict[str, Bucket]] = {
        name: {cond: Bucket() for cond in CONDITIONS} for name in matcher_names
    }

    for script_num in SCRIPT_NUMBERS:
        chunks = load_chunks(SCRIPTS_DIR / f"script{script_num}.txt")
        matchers = build_matchers(chunks, threshold, confirm_windows)

        for condition in CONDITIONS:
            audio_path = RAW_DIR / f"script{script_num}_speaker{speaker}_{condition}.wav"
            labels_path = LABELS_DIR / f"script{script_num}_speaker{speaker}_{condition}.json"
            if not audio_path.exists() or not labels_path.exists():
                print(f"[script{script_num}/{condition}] skipped, missing file")
                continue

            print(f"[script{script_num}/{condition}] transcribing...")
            windows = list(transcriber.stream_file(str(audio_path)))
            transitions = json.loads(labels_path.read_text(encoding="utf-8"))["chunk_transitions"]

            for name, matcher in matchers.items():
                matcher.reset()
                pointer = GroundTruthPointer(transitions)
                results[name][condition].add_recording(replay(windows, matcher, pointer))

    return results


def print_report(results: dict[str, dict[str, Bucket]]):
    for name, by_condition in results.items():
        if not any(by_condition[c].windows for c in CONDITIONS):
            continue
        overall = compute_metrics(by_condition[CONDITIONS[0]].merged_with(
            [by_condition[c] for c in CONDITIONS[1:]]
        ))
        print(f"\n{name}")
        print(
            f"  overall       accuracy={overall.accuracy:.3f}  "
            f"false_advance_rate={overall.false_advance_rate:.3f}  "
            f"f1_advance={overall.f1_advance:.3f}  "
            f"latency={_fmt_latency(overall)}  "
            f"({overall.n_windows} windows, {overall.n_transitions} transitions)"
        )
        for condition in CONDITIONS:
            bucket = by_condition[condition]
            if not bucket.windows:
                continue
            metrics = compute_metrics(bucket)
            print(
                f"  {condition:12} accuracy={metrics.accuracy:.3f}  "
                f"false_advance_rate={metrics.false_advance_rate:.3f}  "
                f"f1_advance={metrics.f1_advance:.3f}  "
                f"latency={_fmt_latency(metrics)}  "
                f"({metrics.n_windows} windows, {metrics.n_transitions} transitions)"
            )


def _fmt_latency(metrics: Metrics) -> str:
    if metrics.mean_advance_latency is None:
        return "n/a"
    missed = metrics.n_transitions - metrics.n_latencies_measured
    return f"{metrics.mean_advance_latency:.2f}s" + (f" ({missed} missed)" if missed else "")


def write_csv(results: dict[str, dict[str, Bucket]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matcher", "condition", "n_windows", "accuracy", "false_advance_rate",
        "f1_advance", "n_transitions", "n_latencies_measured", "mean_advance_latency",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, by_condition in results.items():
            overall = by_condition[CONDITIONS[0]].merged_with([by_condition[c] for c in CONDITIONS[1:]])
            rows = [("overall", overall)] + [
                (c, by_condition[c]) for c in CONDITIONS if by_condition[c].windows
            ]
            for condition, bucket in rows:
                m = compute_metrics(bucket)
                writer.writerow({
                    "matcher": name, "condition": condition, "n_windows": m.n_windows,
                    "accuracy": m.accuracy, "false_advance_rate": m.false_advance_rate,
                    "f1_advance": m.f1_advance, "n_transitions": m.n_transitions,
                    "n_latencies_measured": m.n_latencies_measured,
                    "mean_advance_latency": m.mean_advance_latency,
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", default="A")
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument(
        "--confirm-windows", type=int, default=1,
        help="consecutive 'advance' predictions the classifier matcher needs before "
             "committing a chunk transition (hysteresis mitigation for cascading false advances)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "evaluation" / "results.csv")
    args = parser.parse_args()

    results = run_evaluation(args.speaker, args.threshold, args.confirm_windows)
    print_report(results)
    write_csv(results, args.out)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
