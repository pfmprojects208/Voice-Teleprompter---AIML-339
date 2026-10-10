"""Trains the B3 advance/hold classifier described in the README.

Loads the per-window feature rows produced by `dataset/build_dataset.py`
(`data/features/speaker*.csv`), compares logistic regression against a
small random forest under leave-one-script-out cross-validation, and
saves the better candidate - retrained on all available data - as the
B3 matcher's model.

Splitting by script (rather than by row, or even by recording) keeps
every window from a script's three condition recordings on the same
side of the split, so held-out metrics measure generalisation to unseen
script content instead of leaking adjacent windows across train/test.

Both candidates use asymmetric class weighting: a false advance
(predicting advance on a true hold) costs 3x a false hold, per the
README's "Advance classifier" section, so the hold class is weighted 3x.

Usage (from src/evaluation/):
    python train_classifier.py
    python train_classifier.py --out ../../data/models/b3_classifier.joblib
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "data" / "features"
MODELS_DIR = REPO_ROOT / "data" / "models"

FEATURE_COLUMNS = [
    "lexical_current", "semantic_current", "semantic_next", "semantic_delta",
    "time_in_chunk", "tokens_seen_ratio", "lexical_rolling_mean", "asr_confidence",
]

# A false advance (predicted advance, true hold) is 3x costlier than a false
# hold - see README "Advance classifier" section. Weighting the hold class
# (0) 3x makes misclassifying a true hold 3x as costly during training,
# which is what discourages false advances.
HOLD_ADVANCE_CLASS_WEIGHT = {0: 3, 1: 1}

# lexical_current/lexical_rolling_mean separate the classes almost perfectly
# within some leave-one-script-out folds, which pushes LogisticRegression's
# LBFGS solver through large intermediate weight values before it settles -
# cosmetic overflow warnings with no effect on the converged result.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.*")


def load_dataset(features_dir: Path = FEATURES_DIR) -> list[dict]:
    rows = []
    for csv_path in sorted(features_dir.glob("speaker*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        raise FileNotFoundError(f"no speaker*.csv files found in {features_dir}")
    return rows


def split_by_script(rows: list[dict], test_script: str) -> tuple[list[dict], list[dict]]:
    train = [r for r in rows if r["script"] != test_script]
    test = [r for r in rows if r["script"] == test_script]
    if not train:
        raise ValueError("training split is empty")
    if not test:
        raise ValueError(f"no rows found for test script {test_script!r}")
    return train, test


def to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[c]) for c in FEATURE_COLUMNS] for r in rows])
    y = np.array([int(r["advance"]) for r in rows])
    return X, y


def false_advance_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Proportion of true holds (0) misclassified as advance (1)."""
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    holds = tn + fp
    return fp / holds if holds else 0.0


def build_candidates() -> dict[str, Callable[[], Pipeline]]:
    """Returns a factory per candidate so each CV fold trains a fresh model."""
    return {
        "logistic_regression": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight=HOLD_ADVANCE_CLASS_WEIGHT, max_iter=1000)),
        ]),
        "random_forest": lambda: Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=6,
                class_weight=HOLD_ADVANCE_CLASS_WEIGHT, random_state=42,
            )),
        ]),
    }


def evaluate(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "f1_advance": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "false_advance_rate": false_advance_rate(y_test, y_pred),
        "accuracy": float((y_pred == y_test).mean()),
    }


def leave_one_script_out_cv(
    rows: list[dict], candidates: dict[str, Callable[[], Pipeline]]
) -> dict[str, dict]:
    scripts = sorted({r["script"] for r in rows})
    fold_metrics: dict[str, list[dict]] = {name: [] for name in candidates}

    for test_script in scripts:
        train_rows, test_rows = split_by_script(rows, test_script)
        X_train, y_train = to_arrays(train_rows)
        X_test, y_test = to_arrays(test_rows)

        for name, build in candidates.items():
            model = build()
            model.fit(X_train, y_train)
            fold_metrics[name].append(evaluate(model, X_test, y_test))

    return {
        name: {
            metric: float(np.mean([fold[metric] for fold in folds]))
            for metric in ("f1_advance", "false_advance_rate", "accuracy")
        }
        for name, folds in fold_metrics.items()
    }


def select_best(cv_results: dict[str, dict]) -> str:
    """Lowest false-advance rate wins - advancing too early is the
    catastrophic error per README - ties broken by f1 on the advance class."""
    return min(
        cv_results, key=lambda name: (cv_results[name]["false_advance_rate"], -cv_results[name]["f1_advance"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--out", type=Path, default=MODELS_DIR / "b3_classifier.joblib")
    args = parser.parse_args()

    rows = load_dataset(args.features_dir)
    print(f"Loaded {len(rows)} rows from {args.features_dir}")

    candidates = build_candidates()
    cv_results = leave_one_script_out_cv(rows, candidates)

    print("\nLeave-one-script-out cross-validation:")
    for name, metrics in cv_results.items():
        print(
            f"  {name:20} f1_advance={metrics['f1_advance']:.3f}  "
            f"false_advance_rate={metrics['false_advance_rate']:.3f}  "
            f"accuracy={metrics['accuracy']:.3f}"
        )

    best_name = select_best(cv_results)
    print(f"\nSelected: {best_name}")

    final_model = candidates[best_name]()
    X, y = to_arrays(rows)
    final_model.fit(X, y)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"name": best_name, "model": final_model, "features": FEATURE_COLUMNS}, args.out)
    print(f"Retrained {best_name} on all {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
