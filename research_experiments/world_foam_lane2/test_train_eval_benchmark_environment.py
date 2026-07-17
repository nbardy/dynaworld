from __future__ import annotations

import unittest
from unittest import mock

import train_eval_owner_run_tape as train_eval


HARD_KEYWORDS = ("pytest", "torch", "metal", "mps")


def _blocks(command: str, pcpu: float) -> bool:
    return train_eval._benchmark_process_blocks_promotion(
        command=command,
        pcpu=pcpu,
        blocking_cpu_threshold=5.0,
        general_blocking_cpu_threshold=75.0,
        hard_keywords=HARD_KEYWORDS,
    )


def _reason(command: str, pcpu: float) -> str | None:
    return train_eval._benchmark_process_block_reason(
        command=command,
        pcpu=pcpu,
        blocking_cpu_threshold=5.0,
        general_blocking_cpu_threshold=75.0,
        hard_keywords=HARD_KEYWORDS,
    )


class BenchmarkEnvironmentTests(unittest.TestCase):
    def test_ps_parser_preserves_runtime_state_fields(self) -> None:
        parsed = train_eval._parse_benchmark_ps_line(
            "7002 6978 R 02:16:39 193.9 2.1 python train_node_curve_program_flow_v2.py"
        )

        self.assertEqual(parsed["pid"], 7002)
        self.assertEqual(parsed["ppid"], 6978)
        self.assertEqual(parsed["stat"], "R")
        self.assertEqual(parsed["elapsed"], "02:16:39")
        self.assertEqual(parsed["pcpu"], 193.9)
        self.assertEqual(parsed["command"], "python train_node_curve_program_flow_v2.py")

    def test_ps_parser_keeps_legacy_fixture_shape_supported(self) -> None:
        parsed = train_eval._parse_benchmark_ps_line("7002 6978 193.9 2.1 python train.py")

        self.assertEqual(parsed["pid"], 7002)
        self.assertEqual(parsed["stat"], "")
        self.assertEqual(parsed["elapsed"], "")
        self.assertEqual(parsed["pcpu"], 193.9)

    def test_keyword_matching_avoids_substring_false_positive(self) -> None:
        self.assertFalse(train_eval._benchmark_keyword_matches("/Steam Helper --max-uploads=5", "mps"))
        self.assertTrue(train_eval._benchmark_keyword_matches("/site-packages/torch/lib/libtorch.dylib", "torch"))
        self.assertTrue(train_eval._benchmark_keyword_matches("/csrc/metal/world_foam_lane2_metal.mm", "metal"))

    def test_idle_pytest_wrapper_does_not_block_promotion(self) -> None:
        self.assertFalse(_blocks("uv run python -m pytest tests/", 0.0))

    def test_high_cpu_pytest_still_blocks_promotion(self) -> None:
        self.assertTrue(_blocks("/opt/homebrew/bin/python -m pytest tests/", 97.5))

    def test_medium_cpu_python_process_still_blocks_promotion(self) -> None:
        self.assertTrue(_blocks("python train_node_curve_program_flow_v2.py", 12.0))
        self.assertEqual(_reason("python train_node_curve_program_flow_v2.py", 12.0), "high_cpu")

    def test_low_cpu_torch_or_mps_process_still_blocks_promotion(self) -> None:
        self.assertTrue(_blocks("/site-packages/torch/lib/libtorch.dylib", 0.0))
        self.assertTrue(_blocks("python src/train/train.py --device mps", 0.0))

    def test_idle_metal_compiler_service_does_not_block_promotion(self) -> None:
        command = (
            "/System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/"
            "MTLCompilerService.xpc/Contents/MacOS/MTLCompilerService"
        )
        self.assertFalse(_blocks(command, 0.0))
        self.assertTrue(_blocks(command, 95.0))

    def test_low_cpu_monitor_wrappers_do_not_block_promotion(self) -> None:
        for command in (
            "SCREEN -dmS btc15m_toto_allfold_policy_overnight_190646 zsh -lc cd /Users/nicholasbardy/git/ai_trader && uv run python scripts/run_btc15m_overnight_shadow_monitor.py --run-id btc15m_toto_allfold_policy_mps_probe",
            "login -pflq nicholasbardy /bin/zsh -lc cd /Users/nicholasbardy/git/ai_trader && uv run python scripts/run_btc15m_overnight_shadow_monitor.py --run-id btc15m_toto_allfold_policy",
            "uv run python scripts/run_btc15m_overnight_shadow_monitor.py --run-id btc15m_toto_allfold_policy_overnight_20260518T190646Z --duration-hours 8 --tag mps",
            "rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_clean_mps_gate_when_ready.py --execute --wait-ready-timeout-s 28800 --summary-json research_experiments/world_foam_lane2/results/clean_mps_wait.json",
            "rtk sh -lc PYTHONPATH=research_experiments/world_foam_lane2 .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py --run-id real32 --max-worldfoam-attempts 2 --max-star-attempts 2",
        ):
            self.assertFalse(_blocks(command, 0.0))

    def test_high_cpu_monitor_wrapper_still_blocks_promotion(self) -> None:
        self.assertTrue(
            _blocks(
                "uv run python scripts/run_btc15m_overnight_shadow_monitor.py --run-id btc15m_toto_allfold_policy_overnight",
                97.0,
            )
        )

    def test_periodic_mps_export_monitor_blocks_even_when_idle(self) -> None:
        command = (
            "/usr/bin/SCREEN -dmS toto_floor001_guardaligned zsh -lc "
            "cd /Users/nicholasbardy/git/ai_trader; "
            "uv run python scripts/run_btc15m_overnight_shadow_monitor.py "
            "--run-id btc15m_toto_context64 --duration-hours 12 "
            "--toto-export-device mps --toto-export-with-runtime-deps"
        )

        self.assertTrue(_blocks(command, 0.0))
        self.assertEqual(_reason(command, 0.0), "periodic_mps_exporter")

    def test_high_cpu_non_keyword_process_blocks_promotion(self) -> None:
        command = "Cursor Helper (Plugin): extension-host (user) empty [1-1]"

        self.assertFalse(_blocks(command, 50.0))
        self.assertTrue(_blocks(command, 300.0))
        self.assertEqual(_reason(command, 300.0), "high_cpu_general")

    def test_captured_blockers_keep_long_monitor_command_flags(self) -> None:
        long_command = (
            "123 1 0.0 0.1 "
            + "uv run python scripts/run_btc15m_overnight_shadow_monitor.py "
            + "--run-id "
            + "x" * 260
            + " --toto-export-device mps --toto-export-with-runtime-deps"
        )
        with mock.patch.object(
            train_eval.subprocess,
            "run",
            return_value=mock.Mock(stdout=long_command + "\n"),
        ):
            environment = train_eval._capture_benchmark_environment()

        self.assertEqual(environment["status"], "contended")
        blocker = environment["blocking_processes"][0]
        self.assertEqual(blocker["block_reason"], "periodic_mps_exporter")
        self.assertLessEqual(len(blocker["command"]), train_eval.BENCHMARK_PROCESS_COMMAND_LIMIT)
        self.assertIn("--toto-export-device", blocker["command"])
        self.assertIn("--toto-export-with-runtime-deps", blocker["command"])

    def test_capture_includes_high_cpu_non_keyword_process(self) -> None:
        ps_stdout = "\n".join(
            [
                "111 1 R 00:42:03 325.0 4.8 Cursor Helper (Plugin): extension-host (user) empty [1-1]",
                "222 1 50.0 0.1 /Applications/SomeIdleLookingApp.app/Contents/MacOS/App",
                "333 1 0.0 0.1 /usr/bin/true",
            ]
        )
        with mock.patch.object(
            train_eval.subprocess,
            "run",
            return_value=mock.Mock(stdout=ps_stdout + "\n"),
        ):
            environment = train_eval._capture_benchmark_environment()

        self.assertEqual(environment["status"], "contended")
        self.assertEqual(environment["blocking_processes"][0]["pid"], 111)
        self.assertEqual(environment["blocking_processes"][0]["stat"], "R")
        self.assertEqual(environment["blocking_processes"][0]["elapsed"], "00:42:03")
        self.assertEqual(environment["blocking_processes"][0]["block_reason"], "high_cpu_general")
        self.assertEqual(
            environment["blocking_processes"][0]["command"],
            "Cursor Helper (Plugin): extension-host (user) empty [1-1]",
        )

    def test_capture_preserves_total_blocker_count_when_sample_is_capped(self) -> None:
        ps_stdout = "\n".join(
            f"{100 + index} 1 {90.0 + index:.1f} 0.1 python worker_{index}.py"
            for index in range(40)
        )
        with mock.patch.object(
            train_eval.subprocess,
            "run",
            return_value=mock.Mock(stdout=ps_stdout + "\n"),
        ):
            environment = train_eval._capture_benchmark_environment()

        self.assertEqual(environment["status"], "contended")
        self.assertEqual(environment["blocking_process_count"], 40)
        self.assertEqual(environment["contending_process_count"], 40)
        self.assertEqual(environment["process_sample_limit"], train_eval.BENCHMARK_PROCESS_SAMPLE_LIMIT)
        self.assertEqual(len(environment["blocking_processes"]), train_eval.BENCHMARK_PROCESS_SAMPLE_LIMIT)
        self.assertEqual(len(environment["contending_processes"]), train_eval.BENCHMARK_PROCESS_SAMPLE_LIMIT)
        self.assertEqual(environment["blocking_processes"][0]["pid"], 139)

    def test_capture_ignores_current_process_ancestor_chain(self) -> None:
        ps_stdout = "\n".join(
            [
                (
                    "20 1 0.0 0.0 "
                    "rtk sh -lc PYTHONPATH=src/train .venv/bin/python "
                    "research_experiments/world_foam_lane2/train_eval_owner_run_tape.py "
                    "--config src/train_configs/local_mac_powerfoam_metal_smoke.jsonc"
                ),
                (
                    "30 20 0.0 0.0 "
                    "/bin/zsh -lc PYTHONPATH=src/train .venv/bin/python "
                    "research_experiments/world_foam_lane2/train_eval_owner_run_tape.py "
                    "--config src/train_configs/local_mac_powerfoam_metal_smoke.jsonc"
                ),
                (
                    "40 30 0.0 0.1 "
                    "/opt/homebrew/bin/python "
                    "research_experiments/world_foam_lane2/train_eval_owner_run_tape.py "
                    "--benchmark-environment-check-only"
                ),
            ]
        )
        with (
            mock.patch.object(train_eval.os, "getpid", return_value=40),
            mock.patch.object(train_eval.os, "getppid", return_value=30),
            mock.patch.object(
                train_eval.subprocess,
                "run",
                return_value=mock.Mock(stdout=ps_stdout + "\n"),
            ),
        ):
            environment = train_eval._capture_benchmark_environment()

        self.assertEqual(environment["status"], "ok")
        self.assertEqual(environment["blocking_processes"], [])
        self.assertEqual(environment["background_processes"], [])

    def test_background_environment_does_not_block_promotion(self) -> None:
        environment = {
            "status": "background",
            "blocking_processes": [],
            "background_processes": [{"pid": 5391, "pcpu": 0.1, "command": "python -m sky.server.server"}],
        }

        self.assertFalse(train_eval._benchmark_environment_blocks_promotion(environment))

    def test_contended_environment_blocks_promotion(self) -> None:
        environment = {
            "status": "contended",
            "blocking_processes": [{"pid": 147, "pcpu": 98.0, "command": "python train.py"}],
            "background_processes": [],
        }

        self.assertTrue(train_eval._benchmark_environment_blocks_promotion(environment))

    def test_unchecked_environment_blocks_promotion(self) -> None:
        environment = {
            "status": "unchecked",
            "error": "ps failed",
            "blocking_processes": [],
            "background_processes": [],
        }

        self.assertTrue(train_eval._benchmark_environment_blocks_promotion(environment))

    def test_merge_marks_any_contended_snapshot_as_contended(self) -> None:
        merged = train_eval._merge_benchmark_environments(
            {"status": "background", "background_processes": [{"pid": 1}]},
            {"status": "contended", "blocking_processes": [{"pid": 2}]},
        )

        self.assertEqual(merged["status"], "contended")

    def test_post_run_settle_can_clear_transient_mtl_compiler_only_snapshot(self) -> None:
        mtl_process = {
            "pid": 44,
            "pcpu": 75.0,
            "command": "/System/Library/Frameworks/Metal.framework/Versions/A/XPCServices/MTLCompilerService.xpc/Contents/MacOS/MTLCompilerService",
        }
        with (
            mock.patch.object(
                train_eval,
                "_capture_benchmark_environment",
                side_effect=[
                    {"status": "contended", "blocking_processes": [mtl_process]},
                    {"status": "background", "blocking_processes": [], "background_processes": [{"pid": 1}]},
                ],
            ),
            mock.patch.object(train_eval.time, "sleep") as sleep_mock,
        ):
            merged = train_eval._merge_benchmark_environments_with_optional_settle(
                {"status": "background"},
                settle_s=0.5,
            )

        self.assertEqual(merged["status"], "background")
        self.assertTrue(merged["transient_mtl_compiler_settled"])
        self.assertEqual(merged["end_immediate"]["blocking_processes"][0]["pid"], 44)
        sleep_mock.assert_called_once_with(0.5)

    def test_post_run_settle_does_not_hide_python_blocker(self) -> None:
        with (
            mock.patch.object(
                train_eval,
                "_capture_benchmark_environment",
                return_value={
                    "status": "contended",
                    "blocking_processes": [{"pid": 55, "pcpu": 40.0, "command": "python train.py --device mps"}],
                },
            ),
            mock.patch.object(train_eval.time, "sleep") as sleep_mock,
        ):
            merged = train_eval._merge_benchmark_environments_with_optional_settle(
                {"status": "background"},
                settle_s=0.5,
            )

        self.assertEqual(merged["status"], "contended")
        self.assertNotIn("end_immediate", merged)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
