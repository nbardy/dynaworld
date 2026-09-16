"""Protect the user's small-Mac budget and fail closed on new memory pressure."""
from copy import deepcopy
import subprocess

import pytest

from config_utils import load_config_file
from paper_local_resources import (
    GIB, check_running_limits, normalize_local_resources, process_tree_rss,
)
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite import run_unified_paper_ablation as runner


def raw_protocol():
    return load_config_file(
        runner.ROOT / "src/train_configs/paper_protocols/coffee_martini_local_playground.jsonc"
    )


def host():
    return {"platform": "darwin", "available_memory_bytes": 6 * GIB,
            "disk_free_bytes": 20 * GIB, "swap_used_bytes": 7 * GIB,
            "load_1m_per_logical_cpu": 0.2}


def test_playground_accepts_idle_host_with_old_swap_but_publication_does_not():
    protocol = resolve_paper_training_protocol(raw_protocol())
    runner.require_live_resources(host(), protocol)
    with pytest.raises(RuntimeError, match="swap_used_bytes"):
        runner.require_live_resources(host())
    assert "local_resources" not in resolve_paper_training_protocol(
        load_config_file(runner.DEFAULT_PROTOCOL)
    ).as_dict()


@pytest.mark.parametrize("key,value", [
    ("process_tree_rss_limit_bytes", 6 * GIB), ("disk_budget_bytes", 17 * GIB),
])
def test_oversized_playground_budget_is_rejected(key, value):
    raw = deepcopy(raw_protocol()["local_resources"])
    raw[key] = value
    with pytest.raises(ValueError, match="8 GiB RAM and 16 GiB disk"):
        normalize_local_resources(raw)


@pytest.mark.parametrize("change,error", [
    ({"tree_rss": 4 * GIB}, "process-tree RSS"),
    ({"output_bytes": 3 * GIB}, "output disk"),
    ({"snapshot": {**host(), "swap_used_bytes": 8 * GIB}}, "swap growth"),
    ({"snapshot": {**host(), "available_memory_bytes": GIB}}, "memory headroom"),
])
def test_running_job_stops_when_a_budget_is_exceeded(change, error):
    args = dict(tree_rss=GIB, output_bytes=1024, initial_swap=7 * GIB, snapshot=host())
    check_running_limits(raw_protocol()["local_resources"], **args)
    with pytest.raises(RuntimeError, match=error):
        check_running_limits(raw_protocol()["local_resources"], **{**args, **change})


def test_memory_measurement_includes_grandchildren_and_launcher(monkeypatch):
    monkeypatch.setattr("paper_local_resources.os.getpid", lambda: 10)
    # Root and grandchildren are deliberately unordered, with an unrelated job.
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k:
                        "12 11 300\n99 1 9000\n10 1 100\n11 20 200\n20 10 400\n")
    assert process_tree_rss(20) == 1000 * 1024
