from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.evaluation.train_classifier import (
    FEATURE_COLUMNS,
    HOLD_ADVANCE_CLASS_WEIGHT,
    build_candidates,
    false_advance_rate,
    leave_one_script_out_cv,
    select_best,
    split_by_script,
    to_arrays,
)


def _row(script: str, advance: int, lexical: float) -> dict:
    row = {c: "0.0" for c in FEATURE_COLUMNS}
    row.update(script=script, advance=str(advance), lexical_current=str(lexical))
    return row


# Three synthetic scripts, each with both classes, separable on lexical_current
# alone so every fold's model fits deterministically and fast.
SYNTHETIC_ROWS = [
    _row(script, advance, lexical)
    for script in ("1", "2", "3")
    for advance, lexical in [(0, 10.0), (0, 15.0), (0, 20.0), (1, 80.0), (1, 85.0), (1, 90.0)]
]


def test_split_by_script_separates_rows_by_script_field():
    train, test = split_by_script(SYNTHETIC_ROWS, test_script="2")

    assert all(r["script"] != "2" for r in train)
    assert all(r["script"] == "2" for r in test)
    assert len(train) + len(test) == len(SYNTHETIC_ROWS)


def test_split_by_script_rejects_unknown_test_script():
    with pytest.raises(ValueError, match="no rows"):
        split_by_script(SYNTHETIC_ROWS, test_script="9")


def test_to_arrays_matches_feature_column_order_and_label():
    rows = [_row("1", advance=1, lexical=42.0)]

    X, y = to_arrays(rows)

    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert X[0, FEATURE_COLUMNS.index("lexical_current")] == 42.0
    assert y[0] == 1


def test_false_advance_rate_counts_only_false_positives_among_true_holds():
    import numpy as np

    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    assert false_advance_rate(y_true, y_pred) == pytest.approx(2 / 3)


def test_false_advance_rate_is_zero_with_no_true_holds():
    import numpy as np

    y_true = np.array([1, 1])
    y_pred = np.array([1, 0])

    assert false_advance_rate(y_true, y_pred) == 0.0


def test_hold_class_is_weighted_three_times_advance():
    assert HOLD_ADVANCE_CLASS_WEIGHT == {0: 3, 1: 1}


def test_leave_one_script_out_cv_reports_metrics_for_every_candidate():
    results = leave_one_script_out_cv(SYNTHETIC_ROWS, build_candidates())

    assert set(results) == {"logistic_regression", "random_forest"}
    for metrics in results.values():
        assert set(metrics) == {"f1_advance", "false_advance_rate", "accuracy"}
        for value in metrics.values():
            assert 0.0 <= value <= 1.0


def test_select_best_prefers_lower_false_advance_rate():
    results = {
        "a": {"false_advance_rate": 0.1, "f1_advance": 0.5},
        "b": {"false_advance_rate": 0.2, "f1_advance": 0.9},
    }

    assert select_best(results) == "a"


def test_select_best_breaks_ties_with_higher_f1_advance():
    results = {
        "a": {"false_advance_rate": 0.1, "f1_advance": 0.5},
        "b": {"false_advance_rate": 0.1, "f1_advance": 0.9},
    }

    assert select_best(results) == "b"
