from alpha_evolve.evolver.agreement import build_selection_report
from alpha_evolve.evolver.codex_command import build_codex_exec_command


def test_dual_agreement_beats_largest_wrong_cluster():
    matrix = {
        "task_id": "largest_cluster_trap",
        "selection_budget": 1,
        "probe_ids": ["p1", "p2", "p3"],
        "primary_metric": {"name": "backward_ms", "direction": "min"},
        "candidates": [
            {
                "candidate_id": "wrong_a",
                "visible_pass": True,
                "probe_results": {"p1": True, "p2": False, "p3": False},
                "metrics": {"backward_ms": 1.0},
                "hidden_pass": False,
            },
            {
                "candidate_id": "wrong_b",
                "visible_pass": True,
                "probe_results": {"p1": True, "p2": False, "p3": False},
                "metrics": {"backward_ms": 2.0},
                "hidden_pass": False,
            },
            {
                "candidate_id": "wrong_c",
                "visible_pass": True,
                "probe_results": {"p1": True, "p2": False, "p3": False},
                "metrics": {"backward_ms": 3.0},
                "hidden_pass": False,
            },
            {
                "candidate_id": "correct",
                "visible_pass": True,
                "probe_results": {"p1": True, "p2": True, "p3": True},
                "metrics": {"backward_ms": 100.0},
                "hidden_pass": True,
            },
        ],
    }

    report = build_selection_report(matrix)

    assert report["selected_candidate_ids"] == ["correct"]
    assert report["selected_hidden_success"] is True
    assert report["oracle_hidden_success"] is True
    assert report["ranker_gap"] == 0
    assert report["visible_false_positive_rate"] == 0.75


def test_selection_reports_ranker_gap_when_selector_misses_oracle():
    matrix = {
        "task_id": "ranker_gap",
        "selection_budget": 1,
        "probe_ids": ["p1", "p2"],
        "candidates": [
            {
                "candidate_id": "wrong_many_tests",
                "visible_pass": True,
                "probe_results": {"p1": True, "p2": True},
                "hidden_pass": False,
            },
            {
                "candidate_id": "correct_fewer_tests",
                "visible_pass": True,
                "probe_results": {"p1": True, "p2": False},
                "hidden_pass": True,
            },
        ],
    }

    report = build_selection_report(matrix)

    assert report["selected_candidate_ids"] == ["wrong_many_tests"]
    assert report["selected_hidden_success"] is False
    assert report["oracle_hidden_success"] is True
    assert report["ranker_gap"] == 1


def test_codex_exec_command_keeps_prompt_positional():
    command = build_codex_exec_command(
        worktree="/tmp/worktree",
        prompt="Improve the kernel.",
        final_message_path="/tmp/candidate/final.md",
        model="gpt-5.4",
    )

    assert command[:2] == ["codex", "exec"]
    assert "--cd" in command
    assert "-p" not in command
    assert command[-1] == "Improve the kernel."
