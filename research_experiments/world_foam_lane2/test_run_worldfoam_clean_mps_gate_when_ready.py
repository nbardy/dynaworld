from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_worldfoam_clean_mps_gate_when_ready as ready_gate


def _goal_report(*, ready: bool, command: list[str] | None = None) -> dict[str, object]:
    return {
        "status": "incomplete_ready_for_clean_mps_gate" if ready else "blocked_external_environment",
        "objective_complete": False,
        "clean_mps_rerun_plan": {
            "ready_to_run_now": ready,
            "blocking_conditions": [] if ready else ["current_benchmark_environment_contended"],
            "wait_reason": None if ready else "current benchmark environment probe must report ok/background",
            "run_after_estimated_done_at": "2026-05-21T12:16:25+07:00",
            "run_after_estimated_done_at_scope": "live_blocker_diagnosis_only",
            "run_after_estimated_done_at_requires_reprobe": True,
            "live_max_estimated_remaining_s_by_category": {}
            if ready
            else {"ai_trader_toto_mps_exporter": 23801.0},
            "current_benchmark_environment_status": "ok" if ready else "contended",
            "current_benchmark_environment_blocking_kind_counts": {}
            if ready
            else {"font_maker_random_stroke_train": 1},
            "current_benchmark_environment_blocking_reason_counts": {} if ready else {"high_cpu": 1},
            "current_benchmark_environment_blocking_screen_session_names": []
            if ready
            else ["toto_floor001_postfix_20260520T171609Z"],
            "current_benchmark_environment_blocking_process_count": 0 if ready else 1,
            "current_benchmark_environment_manual_next_actions": []
            if ready
            else ["wait for font_maker random-stroke training to finish or pause it"],
            "current_benchmark_environment_blocking_process_sample": []
            if ready
            else [
                {
                    "pid": 13496,
                    "block_reason": "high_cpu",
                    "command": "python /Users/nicholasbardy/git/font_maker/train_node_curve_program_flow_v2.py",
                }
            ],
            "live_blocking_category_counts": {},
            "live_blocking_screen_session_names": []
            if ready
            else ["toto_floor001_postfix_20260520T171609Z"],
            "live_recent_output_category_counts": {},
            "command": command
            if command is not None
            else ["rtk", "env", "python", "gate.py", "--verify-result"],
            "acceptance_gate": "do not promote until verifier reports ok",
            "embedded_result_verification": True,
            "acceptance_verifier_required_status": "ok",
            "acceptance_verifier_command_template": [
                "rtk",
                "env",
                "python",
                "verify_worldfoam_next_mps_candidate_result.py",
                "<launch_summary_json>",
            ],
        },
    }


class RunWorldFoamCleanMpsGateWhenReadyTests(unittest.TestCase):
    def test_not_ready_fails_closed_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            goal = Path(tmpdir) / "goal.json"
            with mock.patch.object(ready_gate.refresh_mod, "refresh", return_value=_goal_report(ready=False)):
                with mock.patch.object(ready_gate.subprocess, "run") as subprocess_run:
                    returncode, payload = ready_gate.run_gate_when_ready(
                        execute=True,
                        summary_json=summary,
                        goal_state_json=goal,
                    )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 2)
        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(written["status"], "not_ready")
        self.assertEqual(written["blocking_conditions"], ["current_benchmark_environment_contended"])
        self.assertEqual(written["current_benchmark_environment_blocking_process_count"], 1)
        self.assertEqual(written["current_benchmark_environment_blocking_reason_counts"], {"high_cpu": 1})
        self.assertEqual(written["current_benchmark_environment_blocking_process_sample"][0]["pid"], 13496)
        self.assertIn(
            "wait for font_maker random-stroke training to finish or pause it",
            written["current_benchmark_environment_manual_next_actions"],
        )
        subprocess_run.assert_not_called()

    def test_stdout_summary_omits_process_sample_but_keeps_blocker_counts(self) -> None:
        payload = ready_gate._ready_gate_payload(
            report=_goal_report(ready=False),
            execute=True,
            goal_state_json=Path("/tmp/goal.json"),
        )

        summary = ready_gate._stdout_payload(payload, print_payload="summary")

        self.assertEqual(summary["status"], "not_ready")
        self.assertEqual(summary["current_benchmark_environment_blocking_kind_counts"], {
            "font_maker_random_stroke_train": 1
        })
        self.assertEqual(
            summary["live_max_estimated_remaining_s_by_category"],
            {"ai_trader_toto_mps_exporter": 23801.0},
        )
        self.assertEqual(
            summary["current_benchmark_environment_blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )
        self.assertEqual(
            summary["live_blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )
        self.assertTrue(summary["embedded_result_verification"])
        self.assertEqual(summary["acceptance_verifier_required_status"], "ok")
        self.assertEqual(
            summary["acceptance_verifier_command_template"][-2:],
            ["verify_worldfoam_next_mps_candidate_result.py", "<launch_summary_json>"],
        )
        self.assertEqual(summary["current_benchmark_environment_blocking_process_count"], 1)
        self.assertNotIn("current_benchmark_environment_blocking_process_sample", summary)
        self.assertIs(ready_gate._stdout_payload(payload, print_payload="full"), payload)

    def test_ready_dry_run_writes_command_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(ready_gate.refresh_mod, "refresh", return_value=_goal_report(ready=True)):
                with mock.patch.object(ready_gate.subprocess, "run") as subprocess_run:
                    returncode, payload = ready_gate.run_gate_when_ready(
                        execute=False,
                        summary_json=summary,
                    )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 0)
        self.assertEqual(payload["status"], "ready_dry_run")
        self.assertEqual(written["command"], ["rtk", "env", "python", "gate.py", "--verify-result"])
        subprocess_run.assert_not_called()

    def test_ready_execute_launches_embedded_command_and_refreshes_after(self) -> None:
        reports = [
            _goal_report(ready=True, command=["rtk", "env", "python", "gate.py", "--verify-result"]),
            {"status": "complete", "objective_complete": True, "clean_mps_rerun_plan": {"ready_to_run_now": True}},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(ready_gate.refresh_mod, "refresh", side_effect=reports) as refresh:
                with mock.patch.object(
                    ready_gate.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
                ) as subprocess_run:
                    returncode, payload = ready_gate.run_gate_when_ready(
                        execute=True,
                        summary_json=summary,
                    )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 0)
        self.assertEqual(payload["status"], "launch_ok")
        self.assertEqual(payload["post_launch_goal_status"], "complete")
        self.assertTrue(written["post_launch_objective_complete"])
        subprocess_run.assert_called_once()
        self.assertEqual(subprocess_run.call_args.args[0], ["rtk", "env", "python", "gate.py", "--verify-result"])
        self.assertEqual(refresh.call_count, 2)

    def test_ready_execute_fails_if_report_has_no_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                ready_gate.refresh_mod,
                "refresh",
                return_value=_goal_report(ready=True, command=[]),
            ):
                returncode, payload = ready_gate.run_gate_when_ready(
                    execute=True,
                    summary_json=summary,
                )

        self.assertEqual(returncode, 1)
        self.assertEqual(payload["status"], "ready_but_missing_command")

    def test_ready_execute_fails_if_command_lacks_result_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(
                ready_gate.refresh_mod,
                "refresh",
                return_value=_goal_report(ready=True, command=["rtk", "env", "python", "gate.py"]),
            ):
                with mock.patch.object(ready_gate.subprocess, "run") as subprocess_run:
                    returncode, payload = ready_gate.run_gate_when_ready(
                        execute=True,
                        summary_json=summary,
                    )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 1)
        self.assertEqual(payload["status"], "ready_but_unverified_command")
        self.assertIn("clean MPS command must include --verify-result", written["verification_contract_failures"])
        subprocess_run.assert_not_called()

    def test_ready_execute_fails_if_verifier_contract_metadata_is_missing(self) -> None:
        report = _goal_report(ready=True)
        plan = report["clean_mps_rerun_plan"]
        assert isinstance(plan, dict)
        plan["embedded_result_verification"] = False
        plan["acceptance_verifier_required_status"] = "passed"
        plan["acceptance_verifier_command_template"] = ["python", "other.py"]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(ready_gate.refresh_mod, "refresh", return_value=report):
                with mock.patch.object(ready_gate.subprocess, "run") as subprocess_run:
                    returncode, payload = ready_gate.run_gate_when_ready(
                        execute=True,
                        summary_json=summary,
                    )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 1)
        self.assertEqual(payload["status"], "ready_but_unverified_command")
        self.assertIn(
            "clean MPS plan must mark embedded_result_verification=true",
            written["verification_contract_failures"],
        )
        self.assertIn(
            "clean MPS verifier required status must be ok",
            written["verification_contract_failures"],
        )
        self.assertIn(
            "clean MPS verifier command template must call result verifier",
            written["verification_contract_failures"],
        )
        self.assertIn(
            "clean MPS verifier command template must include <launch_summary_json>",
            written["verification_contract_failures"],
        )
        subprocess_run.assert_not_called()

    def test_wait_ready_reprobes_until_ready_before_launch(self) -> None:
        reports = [
            _goal_report(ready=False),
            _goal_report(ready=True, command=["rtk", "env", "python", "gate.py", "--verify-result"]),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(ready_gate.refresh_mod, "refresh", side_effect=reports) as refresh:
                with mock.patch.object(ready_gate.time, "monotonic", side_effect=[0.0, 0.0, 0.0, 0.1]):
                    with mock.patch.object(ready_gate.time, "sleep") as sleep:
                        with mock.patch.object(
                            ready_gate.subprocess,
                            "run",
                            return_value=SimpleNamespace(returncode=0, stdout="ok", stderr=""),
                        ) as subprocess_run:
                            returncode, payload = ready_gate.run_gate_when_ready(
                                execute=True,
                                summary_json=summary,
                                wait_ready_timeout_s=10.0,
                                wait_ready_poll_s=1.0,
                                refresh_after_launch=False,
                            )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 0)
        self.assertEqual(payload["status"], "launch_ok")
        self.assertEqual(payload["ready_refresh_count"], 2)
        self.assertEqual(written["wait_ready_timeout_s"], 10.0)
        refresh.assert_has_calls([mock.call(
            source_json=ready_gate.goal_report.DEFAULT_SOURCE_JSON,
            import_json=ready_gate.goal_report.DEFAULT_IMPORT_JSON,
            smoke_bundle_json=ready_gate.goal_report.DEFAULT_SMOKE_BUNDLE_JSON,
            next_mps_summary_json=None,
            blocker_diagnosis_json=ready_gate.goal_report.DEFAULT_BLOCKER_DIAGNOSIS_JSON,
            out_json=ready_gate.refresh_mod.DEFAULT_GOAL_STATE_JSON,
            recent_seconds=ready_gate.DEFAULT_RECENT_SECONDS,
            max_blocker_diagnosis_age_s=ready_gate.goal_report.DEFAULT_MAX_BLOCKER_DIAGNOSIS_AGE_S,
            probe_current_benchmark_environment=True,
            current_benchmark_environment_config=ready_gate.refresh_mod.launcher.DEFAULT_CONFIG,
            current_benchmark_environment_wait_timeout_s=0.0,
            current_benchmark_environment_wait_poll_s=15.0,
        )] * 2)
        sleep.assert_called_once_with(1.0)
        subprocess_run.assert_called_once()

    def test_wait_ready_timeout_fails_closed_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Path(tmpdir) / "summary.json"
            with mock.patch.object(ready_gate.refresh_mod, "refresh", return_value=_goal_report(ready=False)):
                with mock.patch.object(ready_gate.time, "monotonic", side_effect=[0.0, 2.0, 2.0]):
                    with mock.patch.object(ready_gate.subprocess, "run") as subprocess_run:
                        returncode, payload = ready_gate.run_gate_when_ready(
                            execute=True,
                            summary_json=summary,
                            wait_ready_timeout_s=1.0,
                            wait_ready_poll_s=1.0,
                        )

            written = json.loads(summary.read_text(encoding="utf-8"))

        self.assertEqual(returncode, 2)
        self.assertEqual(payload["status"], "not_ready_timeout")
        self.assertEqual(written["wait_remaining_s"], 0.0)
        subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
