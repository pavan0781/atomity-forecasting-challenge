import pandas as pd
import pytest

from cwfc.data import build_aggregate_targets, chronological_split


def test_aggregate_targets_counts_unique_valid_vms_and_does_not_fill_gaps():
    readings = pd.DataFrame(
        {
            "timestamp": [0, 0, 300, 300],
            "vm_id": ["a", "b", "a", "b"],
            "avg_cpu": [20.0, 40.0, 30.0, None],
        }
    )
    targets = build_aggregate_targets(readings)

    assert targets["active_vm_count"].tolist() == [2.0, 1.0]
    assert targets["aggregate_cpu_utilization"].tolist() == [30.0, 30.0]


def test_aggregate_targets_rejects_duplicate_vm_timestamp():
    readings = pd.DataFrame(
        {
            "timestamp": [0, 0],
            "vm_id": ["a", "a"],
            "avg_cpu": [20.0, 40.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_aggregate_targets(readings)


def test_aggregate_targets_rejects_irregular_timestamp_grid():
    readings = pd.DataFrame(
        {
            "timestamp": [0, 600],
            "vm_id": ["a", "a"],
            "avg_cpu": [20.0, 40.0],
        }
    )

    with pytest.raises(ValueError, match="300-second grid"):
        build_aggregate_targets(readings)


def test_chronological_split_reserves_last_24_rows():
    targets = pd.DataFrame(
        {
            "timestamp": range(134),
            "active_vm_count": range(134),
            "aggregate_cpu_utilization": range(134),
        }
    )
    train, holdout = chronological_split(targets)

    assert len(train) == 110
    assert len(holdout) == 24
    assert holdout["timestamp"].tolist() == list(range(110, 134))
