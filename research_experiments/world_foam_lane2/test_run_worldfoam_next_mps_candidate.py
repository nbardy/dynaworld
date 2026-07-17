from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import run_worldfoam_next_mps_candidate as launcher


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _readiness(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ok",
        "next_mps_candidate": "legacy_pixel_mean",
        "ready_for_quiet_mps_quality_speed_run": True,
        "quality_claim": False,
        "speed_claim": False,
        "mps_quality_speed_artifact_required": True,
    }
    payload.update(overrides)
    return payload


class WorldFoamNextMpsCandidateLauncherTests(unittest.TestCase):
    def test_plan_threads_readiness_candidate_into_train_eval_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            args = launcher.parse_args(
                [
                    "--run-id",
                    "unit",
                    "--readiness",
                    str(readiness),
                    "--summary-json",
                    str(tmp / "summary.json"),
                ]
            )

            summary = launcher.build_summary(args)

        self.assertEqual(summary["status"], "planned")
        command = summary["train_eval_command"]
        self.assertIn("--site-initialization", command)
        self.assertEqual(command[command.index("--site-initialization") + 1], "legacy_pixel_mean")
        self.assertIn("--require-benchmark-environment-ok", command)
        self.assertIn("--experimental-native-owner-run-cutwalk-delta", command)
        verifier_command = summary["result_verifier_command"]
        self.assertIn("verify_worldfoam_next_mps_candidate_result.py", verifier_command[1])
        self.assertEqual(verifier_command[-1], str(tmp / "summary.json"))
        self.assertEqual(summary["history_jsonl"], str(tmp / "summary.history.jsonl"))

    def test_plan_normalizes_relative_output_paths_before_verification(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            args = launcher.parse_args(
                [
                    "--run-id",
                    "unit",
                    "--readiness",
                    str(readiness),
                    "--summary-json",
                    "research_experiments/world_foam_lane2/results/unit_summary.json",
                    "--out-json",
                    "research_experiments/world_foam_lane2/results/unit.worldfoam.json",
                ]
            )

            summary = launcher.build_summary(args)

        planned = Path(summary["planned_worldfoam_artifact"])
        summary_json = Path(summary["summary_json"])
        command = summary["train_eval_command"]
        verifier_command = summary["result_verifier_command"]
        self.assertTrue(planned.is_absolute())
        self.assertTrue(summary_json.is_absolute())
        self.assertEqual(command[command.index("--out-json") + 1], str(planned))
        self.assertEqual(verifier_command[-1], str(summary_json))

    def test_plan_fails_when_readiness_has_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness(quality_claim=True))
            args = launcher.parse_args(["--readiness", str(readiness), "--summary-json", str(tmp / "summary.json")])

            summary = launcher.build_summary(args)

        self.assertEqual(summary["status"], "readiness_failed")
        self.assertIn("quality_claim=false", summary["failures"][0])
        self.assertIsNone(summary["train_eval_command"])

    def test_plan_mode_writes_summary_without_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        self.assertEqual(payload["status"], "planned")
        self.assertFalse(payload["execute"])
        self.assertFalse((tmp / "summary.history.jsonl").exists())

    def test_execute_preflight_failure_writes_blocker_summary_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"
            preflight_payload = {
                "status": "contended",
                "process_sample_limit": 32,
                "blocking_process_count": 4,
                "contending_process_count": 5,
                "blocking_processes": [
                    {
                        "pid": 123,
                        "ppid": 1,
                        "stat": "S+",
                        "elapsed": "02:16:39",
                        "block_reason": "periodic_mps_exporter",
                        "pcpu": 0.0,
                        "pmem": 0.1,
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --toto-export-device mps",
                    },
                    {
                        "pid": 456,
                        "ppid": 1,
                        "block_reason": "high_cpu",
                        "pcpu": 140.0,
                        "pmem": 1.2,
                        "command": "python train_node_curve_program_flow_v2.py",
                    },
                ],
                "contending_processes": [{"pid": 123}, {"pid": 456}, {"pid": 789}],
            }

            stdout = io.StringIO()
            with mock.patch.object(
                launcher,
                "_run_json_command",
                return_value=(2, preflight_payload, json.dumps(preflight_payload), ""),
            ), mock.patch.object(launcher.subprocess, "run") as train_run, redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                        "--execute",
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            history = [
                json.loads(line)
                for line in (tmp / "summary.history.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rc, 2)
        train_run.assert_not_called()
        self.assertEqual(payload["status"], "preflight_contended")
        self.assertEqual(payload["preflight_benchmark_environment_status"], "contended")
        self.assertEqual(payload["preflight_process_sample_limit"], 32)
        self.assertEqual(payload["preflight_blocking_process_count"], 4)
        self.assertEqual(payload["preflight_blocking_process_sample_count"], 2)
        self.assertEqual(payload["preflight_blocking_process_unlisted_count"], 2)
        self.assertEqual(payload["preflight_contending_process_count"], 5)
        self.assertEqual(payload["preflight_contending_process_sample_count"], 3)
        self.assertEqual(payload["preflight_contending_process_unlisted_count"], 2)
        self.assertEqual(payload["preflight_blocking_reasons"], ["high_cpu", "periodic_mps_exporter"])
        self.assertEqual(payload["preflight_stability_samples_completed"], 1)
        self.assertFalse(payload["preflight_stability_ok"])
        self.assertEqual(payload["preflight_blocking_processes"][0]["pid"], 123)
        self.assertEqual(payload["preflight_blocking_processes"][0]["stat"], "S+")
        self.assertEqual(payload["preflight_blocking_processes"][0]["elapsed"], "02:16:39")
        self.assertIn("toto-export-device mps", payload["preflight_blocking_processes"][0]["command"])
        blocker_summary = payload["preflight_external_blocker_summary"]
        self.assertTrue(blocker_summary["requires_external_quiet_window"])
        self.assertEqual(
            blocker_summary["blocking_reason_counts"],
            {"high_cpu": 1, "periodic_mps_exporter": 1},
        )
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"high_cpu_external_job": 1, "periodic_mps_exporter": 1},
        )
        self.assertIn(
            "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
            blocker_summary["manual_next_actions"],
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], "preflight_contended")
        self.assertEqual(history[0]["preflight_process_sample_limit"], 32)
        self.assertEqual(history[0]["preflight_blocking_process_count"], 4)
        self.assertEqual(history[0]["preflight_blocking_process_sample_count"], 2)
        self.assertEqual(history[0]["preflight_blocking_process_unlisted_count"], 2)
        self.assertEqual(history[0]["preflight_contending_process_count"], 5)
        self.assertEqual(history[0]["preflight_contending_process_unlisted_count"], 2)
        self.assertEqual(history[0]["preflight_blocking_processes"][0]["pid"], 123)
        self.assertEqual(history[0]["preflight_blocking_processes"][0]["stat"], "S+")
        self.assertEqual(history[0]["preflight_blocking_processes"][0]["elapsed"], "02:16:39")
        self.assertEqual(history[0]["blocking_kind_counts"], {"high_cpu_external_job": 1, "periodic_mps_exporter": 1})
        self.assertEqual(history[0]["blocking_reason_counts"], {"high_cpu": 1, "periodic_mps_exporter": 1})
        self.assertEqual(
            history[0]["preflight_attempts"][0]["blocking_reason_counts"],
            {"high_cpu": 1, "periodic_mps_exporter": 1},
        )
        self.assertEqual(history[0]["preflight_attempts"][0]["preflight_process_sample_limit"], 32)

    def test_preflight_summary_classifies_external_torch_and_mps_blockers(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 10,
                        "block_reason": "keyword:torch",
                        "command": "uv run --with torch python diffusion_auto_research/run_queue.py",
                    },
                    {
                        "pid": 11,
                        "block_reason": "keyword:mps",
                        "command": "python export.py --device mps",
                    },
                    {
                        "pid": 12,
                        "block_reason": "other",
                        "command": "/usr/bin/SCREEN -dmS toto_floor001_postfix_20260520T171609Z zsh -lc python scripts/run_btc15m_overnight_shadow_monitor.py",
                    },
                ],
                "contending_processes": [{"pid": 10}, {"pid": 11}, {"pid": 12}],
            }
        )

        self.assertEqual(summary["preflight_blocking_process_count"], 3)
        self.assertEqual(summary["preflight_blocking_process_sample_count"], 3)
        self.assertEqual(summary["preflight_blocking_process_unlisted_count"], 0)
        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"mps_worker": 1, "periodic_mps_exporter": 1, "torch_worker": 1},
        )
        self.assertIn("wait for or manually pause external torch/MPS workers", blocker_summary["manual_next_actions"])
        self.assertIn(
            "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
            blocker_summary["manual_next_actions"],
        )
        self.assertEqual(
            blocker_summary["blocking_screen_session_names"],
            ["toto_floor001_postfix_20260520T171609Z"],
        )

    def test_preflight_summary_classifies_general_high_cpu_blocker(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 2441,
                        "block_reason": "high_cpu_general",
                        "pcpu": 324.3,
                        "command": "Cursor Helper (Plugin): extension-host",
                    },
                ],
                "contending_processes": [{"pid": 2441}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(blocker_summary["blocking_kind_counts"], {"high_cpu_external_job": 1})
        self.assertEqual(blocker_summary["blocking_reason_counts"], {"high_cpu_general": 1})
        self.assertIn(
            "wait for or manually pause high-CPU external training/export jobs",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_macos_spotlight_indexer(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 559,
                        "block_reason": "high_cpu_general",
                        "pcpu": 111.0,
                        "command": (
                            "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                            "Metadata.framework/Versions/A/Support/mds_stores"
                        ),
                    },
                ],
                "contending_processes": [{"pid": 559}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(blocker_summary["blocking_kind_counts"], {"macos_spotlight_indexer": 1})
        self.assertEqual(blocker_summary["blocking_reason_counts"], {"high_cpu_general": 1})
        self.assertIn(
            "wait for macOS Spotlight indexing to cool below the general CPU threshold",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_font_maker_blockers(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 81304,
                        "block_reason": "high_cpu",
                        "pcpu": 204.2,
                        "command": (
                            "/Users/nicholasbardy/git/font_maker/kernel_font_experiments/"
                            "train_node_curve_program_flow_v2.py cfg.jsonc"
                        ),
                    },
                    {
                        "pid": 54059,
                        "block_reason": "keyword:torch",
                        "pcpu": 0.0,
                        "command": (
                            "uv run --with numpy --with torch --with pillow --with fonttools "
                            "python diffusion_auto_research/run_random_stroke_ablation_queue.py --force"
                        ),
                    },
                ],
                "contending_processes": [{"pid": 81304}, {"pid": 54059}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"font_maker_random_stroke_queue": 1, "font_maker_random_stroke_train": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertNotIn("torch_worker", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for font_maker random-stroke training to finish or pause it",
            blocker_summary["manual_next_actions"],
        )
        self.assertIn(
            "wait for or pause the font_maker random-stroke queue wrapper",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_toto_child_worker_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 99352,
                        "block_reason": "high_cpu",
                        "pcpu": 13.1,
                        "command": (
                            "python -m scripts.snapshot_btc15m_kalshi_live_quotes "
                            "--output-dir logs/btc15m_shadow_overnight/"
                            "btc15m_toto_context64_floor001_postfix/iterations/0211/live_quote_snapshot"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "pcpu": 0.0,
                        "command": (
                            "python scripts/run_btc15m_overnight_shadow_monitor.py "
                            "--run-id btc15m_toto_context64_floor001_postfix"
                        ),
                    },
                ],
                "contending_processes": [{"pid": 99352}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_toto_worker": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader/TOTO monitor child workers to finish",
            blocker_summary["manual_next_actions"],
        )
        self.assertIn(
            "wait for or manually pause periodic ai_trader/TOTO MPS exporter work",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_does_not_classify_btc15m_sft_as_toto_exporter(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 78691,
                        "block_reason": "high_cpu",
                        "pcpu": 78.2,
                        "command": (
                            "python scripts/train_kalshi_btc15m_sft.py --input "
                            "/tmp/pytest-of-nicholasbardy/test_train_and_evaluate_script0/btc15m.csv"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 78691}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_sft": 1, "periodic_mps_exporter": 1},
        )
        self.assertIn(
            "wait for ai_trader BTC15M SFT pytest/training workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_btc15m_sft_shadow_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 89289,
                        "block_reason": "high_cpu",
                        "pcpu": 99.2,
                        "command": (
                            "python -m lean_trade.runners.btc_15m_sft_shadow "
                            "--config configs/lean_btc_15m_calibrated_sft_edge_paper.yaml"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 89289}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_sft_shadow": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader BTC15M SFT shadow workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_btc15m_sft_runtime_parity_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 3028,
                        "block_reason": "high_cpu",
                        "pcpu": 94.3,
                        "command": (
                            "/opt/homebrew/bin/python scripts/check_btc15m_sft_runtime_parity.py "
                            "--checkpoint /tmp/pytest-of-nicholasbardy/checkpoint "
                            "--config /tmp/pytest-of-nicholasbardy/runtime.yaml"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 3028}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_sft_runtime_parity": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader BTC15M SFT runtime-parity workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_btc15m_activation_rl_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 99520,
                        "block_reason": "high_cpu",
                        "pcpu": 89.5,
                        "command": (
                            "python scripts/build_btc15m_activation_rl_dataset.py "
                            "--activation-input /tmp/pytest-of-nicholasbardy/activation_bank.parquet "
                            "--output-path /tmp/pytest-of-nicholasbardy/activation_rl_dataset.parquet"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 99520}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_activation_rl": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader BTC15M activation-RL dataset workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_btc15m_activation_bank_integrity_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 1173,
                        "block_reason": "high_cpu",
                        "pcpu": 84.4,
                        "command": (
                            "/opt/homebrew/bin/python "
                            "/Users/nicholasbardy/git/ai_trader/scripts/verify_btc15m_activation_bank_integrity.py "
                            "--activation-bank "
                            "/private/var/folders/tmp/pytest-of-nicholasbardy/test_activation_bank_integrity1/"
                            "activation_bank.parquet"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 1173}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_activation_bank_integrity": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader BTC15M activation-bank integrity workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_btc15m_imitation_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 91596,
                        "block_reason": "high_cpu",
                        "pcpu": 67.2,
                        "command": (
                            "python scripts/train_kalshi_btc15m_imitation.py --input "
                            "/tmp/pytest-of-nicholasbardy/test_btc15m_imitation_tree_sou0/raw_replay.parquet"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 91596}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_imitation": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader BTC15M imitation pytest/training workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_btc15m_dqn_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 3417,
                        "block_reason": "high_cpu",
                        "pcpu": 99.6,
                        "command": (
                            "python scripts/train_kalshi_btc15m_dqn.py "
                            "--input /tmp/pytest-of-nicholasbardy/raw_replay.parquet "
                            "--sft-dir /tmp/pytest-of-nicholasbardy/missing_sft"
                        ),
                    },
                    {
                        "pid": 54895,
                        "block_reason": "periodic_mps_exporter",
                        "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                    },
                ],
                "contending_processes": [{"pid": 3417}, {"pid": 54895}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(
            blocker_summary["blocking_kind_counts"],
            {"ai_trader_btc15m_dqn": 1, "periodic_mps_exporter": 1},
        )
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for ai_trader BTC15M DQN pytest/training workers to finish",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_classifies_font_maker_standard_glyph_monitor_separately(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 3421,
                        "block_reason": "high_cpu",
                        "pcpu": 9.1,
                        "command": "uv run python scripts/utilities/monitor_standard_glyph_exposure.py --notify",
                    },
                    {
                        "pid": 3422,
                        "block_reason": "high_cpu",
                        "pcpu": 8.8,
                        "command": "python scripts/utilities/monitor_standard_glyph_exposure.py --notify",
                    },
                ],
                "contending_processes": [{"pid": 3421}, {"pid": 3422}],
            }
        )

        blocker_summary = summary["preflight_external_blocker_summary"]
        self.assertEqual(blocker_summary["blocking_kind_counts"], {"font_maker_standard_glyph_monitor": 2})
        self.assertNotIn("high_cpu_external_job", blocker_summary["blocking_kind_counts"])
        self.assertIn(
            "wait for or pause the font_maker standard-glyph monitor",
            blocker_summary["manual_next_actions"],
        )

    def test_preflight_summary_preserves_capped_process_sample_count(self) -> None:
        summary = launcher._summarize_preflight_payload(
            {
                "status": "contended",
                "process_sample_limit": 32,
                "blocking_process_count": 13,
                "contending_process_count": 14,
                "blocking_processes": [
                    {"pid": 1, "block_reason": "high_cpu", "command": "python train.py"},
                    {"pid": 2, "block_reason": "keyword:torch", "command": "uv run --with torch python q.py"},
                ],
                "contending_processes": [{"pid": 1}, {"pid": 2}, {"pid": 3}],
            }
        )

        self.assertEqual(summary["preflight_process_sample_limit"], 32)
        self.assertEqual(summary["preflight_blocking_process_count"], 13)
        self.assertEqual(summary["preflight_blocking_process_sample_count"], 2)
        self.assertEqual(summary["preflight_blocking_process_unlisted_count"], 11)
        self.assertEqual(summary["preflight_contending_process_count"], 14)
        self.assertEqual(summary["preflight_contending_process_sample_count"], 3)
        self.assertEqual(summary["preflight_contending_process_unlisted_count"], 11)

    def test_history_entry_keeps_full_blocker_sample(self) -> None:
        entry = launcher._history_entry(
            {
                "run_id": "unit",
                "status": "preflight_contended",
                "preflight_blocking_processes": [
                    {"pid": 1, "block_reason": "high_cpu", "command": "python a.py"},
                    {"pid": 2, "block_reason": "keyword:torch", "command": "python b.py"},
                    {"pid": 3, "block_reason": "keyword:mps", "command": "python c.py --device mps"},
                    {"pid": 4, "block_reason": "periodic_mps_exporter", "command": "python toto.py"},
                ],
                "preflight_external_blocker_summary": {
                    "blocking_kind_counts": {"high_cpu_external_job": 1},
                    "blocking_reason_counts": {"high_cpu": 1},
                },
            }
        )

        self.assertEqual([process["pid"] for process in entry["preflight_blocking_processes"]], [1, 2, 3, 4])

    def test_execute_requires_all_stability_samples_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"
            clean_payload = {
                "status": "ok",
                "blocking_processes": [],
                "contending_processes": [],
            }

            stdout = io.StringIO()
            train_result = mock.Mock(returncode=0)
            with mock.patch.object(
                launcher,
                "_run_json_command",
                side_effect=[
                    (0, clean_payload, json.dumps(clean_payload), ""),
                    (0, clean_payload, json.dumps(clean_payload), ""),
                    (0, clean_payload, json.dumps(clean_payload), ""),
                ],
            ) as preflight_run, mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=train_result,
            ) as train_run, redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                        "--preflight-stability-samples",
                        "3",
                        "--preflight-stability-interval-s",
                        "0",
                        "--execute",
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            history = [
                json.loads(line)
                for line in (tmp / "summary.history.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rc, 0)
        self.assertEqual(preflight_run.call_count, 3)
        train_run.assert_called_once()
        self.assertEqual(payload["status"], "train_eval_ok")
        self.assertEqual(payload["preflight_stability_samples_requested"], 3)
        self.assertEqual(payload["preflight_stability_samples_completed"], 3)
        self.assertTrue(payload["preflight_stability_ok"])
        self.assertEqual([sample["returncode"] for sample in payload["preflight_samples"]], [0, 0, 0])
        self.assertEqual(payload["preflight_attempt_count"], 1)

    def test_execute_retries_preflight_until_clean_stability_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"
            clean_payload = {
                "status": "ok",
                "blocking_processes": [],
                "contending_processes": [],
            }
            dirty_payload = {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 456,
                        "ppid": 1,
                        "block_reason": "high_cpu",
                        "pcpu": 120.0,
                        "pmem": 1.2,
                        "command": "python unrelated_export.py",
                    }
                ],
                "contending_processes": [{"pid": 456}],
            }

            stdout = io.StringIO()
            train_result = mock.Mock(returncode=0)
            with mock.patch.object(
                launcher,
                "_run_json_command",
                side_effect=[
                    (2, dirty_payload, json.dumps(dirty_payload), ""),
                    (0, clean_payload, json.dumps(clean_payload), ""),
                    (0, clean_payload, json.dumps(clean_payload), ""),
                ],
            ) as preflight_run, mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=train_result,
            ) as train_run, redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                        "--preflight-stability-samples",
                        "2",
                        "--preflight-stability-interval-s",
                        "0",
                        "--preflight-retry-timeout-s",
                        "60",
                        "--preflight-retry-poll-s",
                        "0",
                        "--execute",
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            history = [
                json.loads(line)
                for line in (tmp / "summary.history.jsonl").read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(rc, 0)
        self.assertEqual(preflight_run.call_count, 3)
        train_run.assert_called_once()
        self.assertEqual(payload["status"], "train_eval_ok")
        self.assertEqual(payload["preflight_attempt_count"], 2)
        self.assertFalse(payload["preflight_attempts"][0]["stability_ok"])
        self.assertTrue(payload["preflight_attempts"][1]["stability_ok"])
        self.assertEqual(payload["preflight_stability_samples_completed"], 2)
        self.assertTrue(payload["preflight_stability_ok"])
        self.assertEqual([sample["returncode"] for sample in payload["preflight_samples"]], [0, 0])
        self.assertEqual([row["status"] for row in history], ["preflight_retry_waiting", "train_eval_ok"])
        self.assertEqual(history[0]["preflight_attempt_count"], 1)
        self.assertEqual(history[0]["preflight_blocking_process_count"], 1)
        self.assertFalse(history[0]["preflight_stability_ok"])
        self.assertEqual(history[1]["preflight_attempt_count"], 2)

    def test_execute_verify_result_fails_closed_after_train_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"
            clean_payload = {
                "status": "ok",
                "blocking_processes": [],
                "contending_processes": [],
            }
            verifier_payload = {
                "status": "failed",
                "failures": ["WorldFoam artifact acceptance failed"],
            }

            stdout = io.StringIO()
            train_result = mock.Mock(returncode=0)
            with mock.patch.object(
                launcher,
                "_run_json_command",
                return_value=(0, clean_payload, json.dumps(clean_payload), ""),
            ), mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=train_result,
            ) as train_run, mock.patch.object(
                launcher,
                "_run_result_verifier",
                return_value=(1, verifier_payload, json.dumps(verifier_payload), ""),
            ) as verifier_run, redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                        "--execute",
                        "--verify-result",
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 1)
        train_run.assert_called_once()
        verifier_run.assert_called_once()
        self.assertEqual(payload["train_eval_returncode"], 0)
        self.assertEqual(payload["result_verifier_returncode"], 1)
        self.assertEqual(payload["status"], "result_verification_failed")
        self.assertEqual(payload["result_verifier_payload"], verifier_payload)

    def test_execute_verify_result_preserves_success_after_clean_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"
            clean_payload = {
                "status": "ok",
                "blocking_processes": [],
                "contending_processes": [],
            }
            verifier_payload = {"status": "ok", "failures": []}

            stdout = io.StringIO()
            train_result = mock.Mock(returncode=0)
            with mock.patch.object(
                launcher,
                "_run_json_command",
                return_value=(0, clean_payload, json.dumps(clean_payload), ""),
            ), mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=train_result,
            ), mock.patch.object(
                launcher,
                "_run_result_verifier",
                return_value=(0, verifier_payload, json.dumps(verifier_payload), ""),
            ) as verifier_run, redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                        "--execute",
                        "--verify-result",
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 0)
        verifier_run.assert_called_once()
        self.assertEqual(payload["status"], "train_eval_ok")
        self.assertEqual(payload["result_verifier_returncode"], 0)
        self.assertEqual(payload["result_verifier_payload"], verifier_payload)

    def test_execute_stops_if_later_stability_sample_becomes_contended(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            readiness = _write_json(tmp / "readiness.json", _readiness())
            summary_json = tmp / "summary.json"
            clean_payload = {
                "status": "ok",
                "blocking_processes": [],
                "contending_processes": [],
            }
            dirty_payload = {
                "status": "contended",
                "blocking_processes": [
                    {
                        "pid": 789,
                        "ppid": 20724,
                        "block_reason": "keyword:mps",
                        "pcpu": 0.3,
                        "pmem": 1.0,
                        "command": "uv run scripts/export_btc15m_toto_residual_live_prediction_export.py --device mps",
                    }
                ],
                "contending_processes": [{"pid": 789}],
            }

            stdout = io.StringIO()
            with mock.patch.object(
                launcher,
                "_run_json_command",
                side_effect=[
                    (0, clean_payload, json.dumps(clean_payload), ""),
                    (2, dirty_payload, json.dumps(dirty_payload), ""),
                ],
            ) as preflight_run, mock.patch.object(
                launcher.subprocess,
                "run",
            ) as train_run, redirect_stdout(stdout):
                rc = launcher.main(
                    [
                        "--run-id",
                        "unit",
                        "--readiness",
                        str(readiness),
                        "--summary-json",
                        str(summary_json),
                        "--preflight-stability-samples",
                        "3",
                        "--preflight-stability-interval-s",
                        "0",
                        "--execute",
                    ]
                )

            payload = json.loads(summary_json.read_text(encoding="utf-8"))

        self.assertEqual(rc, 2)
        self.assertEqual(preflight_run.call_count, 2)
        train_run.assert_not_called()
        self.assertEqual(payload["status"], "preflight_contended")
        self.assertEqual(payload["preflight_stability_samples_requested"], 3)
        self.assertEqual(payload["preflight_stability_samples_completed"], 2)
        self.assertFalse(payload["preflight_stability_ok"])
        self.assertEqual([sample["returncode"] for sample in payload["preflight_samples"]], [0, 2])
        self.assertEqual(payload["preflight_blocking_reasons"], ["keyword:mps"])


if __name__ == "__main__":
    unittest.main()
