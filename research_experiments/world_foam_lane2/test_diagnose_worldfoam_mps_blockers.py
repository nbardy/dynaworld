from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import diagnose_worldfoam_mps_blockers as diagnose


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class DiagnoseWorldFoamMpsBlockersTests(unittest.TestCase):
    def test_classifies_toto_monitor_and_detects_recent_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ai_trader = root / "ai_trader"
            output_dir = ai_trader / "logs" / "run"
            recent = output_dir / "events.jsonl"
            recent.parent.mkdir(parents=True)
            recent.write_text("{}\n", encoding="utf-8")
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 10,
                            "ppid": 1,
                            "stat": "S+",
                            "elapsed": "03:12:00",
                            "pcpu": 0.0,
                            "pmem": 0.1,
                            "block_reason": "periodic_mps_exporter",
                            "command": (
                                f"cd {ai_trader} && python scripts/run_btc15m_overnight_shadow_monitor.py "
                                "--output-dir logs/run --duration-hours 12 --toto-export-with-runtime-deps"
                            ),
                        },
                        {
                            "pid": 11,
                            "ppid": 10,
                            "stat": "S+",
                            "elapsed": "03:12:00",
                            "pcpu": 0.0,
                            "pmem": 0.1,
                            "block_reason": "periodic_mps_exporter",
                            "command": (
                                "python scripts/run_btc15m_overnight_shadow_monitor.py "
                                "--output-dir logs/run --duration-hours 12 --toto-export-with-runtime-deps"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary, recent_seconds=60)

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["category_counts"], {"ai_trader_toto_mps_exporter": 2})
        self.assertEqual(report["live_category_counts"], {})
        self.assertEqual(report["live_blocker_count"], 0)
        self.assertEqual(report["recent_output_blocker_count"], 2)
        self.assertEqual(report["live_or_recent_blocker_count"], 2)
        self.assertEqual(report["recent_output_category_counts"], {"ai_trader_toto_mps_exporter": 2})
        self.assertEqual(report["max_estimated_remaining_s_by_category"], {"ai_trader_toto_mps_exporter": 31680.0})
        parent, child = report["blockers"]
        self.assertEqual(parent["cwd"], str(ai_trader))
        self.assertEqual(parent["cwd_source"], "direct")
        self.assertEqual(parent["output_dir"], str(output_dir))
        self.assertEqual(parent["recent_outputs"][0]["path"], "events.jsonl")
        self.assertEqual(parent["declared_duration_hours"], 12.0)
        self.assertEqual(parent["elapsed_s"], 11520.0)
        self.assertEqual(parent["estimated_remaining_s"], 31680.0)
        self.assertEqual(child["cwd"], str(ai_trader))
        self.assertEqual(child["cwd_source"], "parent")
        self.assertEqual(child["output_dir"], str(output_dir))
        self.assertEqual(child["recent_outputs"][0]["path"], "events.jsonl")
        self.assertEqual(child["declared_duration_hours"], 12.0)
        self.assertEqual(child["elapsed_s"], 11520.0)
        self.assertEqual(child["estimated_remaining_s"], 31680.0)

    def test_parses_ps_elapsed_variants(self) -> None:
        self.assertEqual(diagnose._parse_ps_elapsed_seconds("00:01"), 1.0)
        self.assertEqual(diagnose._parse_ps_elapsed_seconds("04:41:37"), 16897.0)
        self.assertEqual(diagnose._parse_ps_elapsed_seconds("1-02:03:04"), 93784.0)
        self.assertIsNone(diagnose._parse_ps_elapsed_seconds("not-time"))

    def test_classifies_font_maker_train_as_active_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 20,
                            "pcpu": 199.0,
                            "pmem": 2.2,
                            "block_reason": "high_cpu",
                            "command": (
                                "/Users/nicholasbardy/git/font_maker/kernel_font_experiments/"
                                "train_node_curve_program_flow_v2.py cfg.jsonc"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"font_maker_random_stroke_train": 1})
        self.assertEqual(report["live_category_counts"], {})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"font_maker_random_stroke_train": 1})
        self.assertFalse(report["blockers"][0]["active_cpu"])
        self.assertTrue(report["blockers"][0]["summary_cpu_active"])

    def test_live_cpu_threshold_separates_active_from_still_blocking(self) -> None:
        process = {
            "pid": 50,
            "ppid": 1,
            "stat": "R",
            "elapsed": "01:00",
            "pcpu": 80.0,
            "pmem": 3.0,
            "block_reason": "high_cpu_general",
            "command": "Codex Helper Renderer",
        }

        cooled = diagnose._diagnose_process(
            process,
            now_s=time.time(),
            recent_seconds=60,
            direct_cwds={},
            parents={},
            live_processes={50: {"pcpu": 20.0, "pmem": 3.0, "stat": "S", "elapsed": "01:10", "command": "Codex"}},
            known_cwds=[],
        )
        hot = diagnose._diagnose_process(
            process,
            now_s=time.time(),
            recent_seconds=60,
            direct_cwds={},
            parents={},
            live_processes={50: {"pcpu": 80.0, "pmem": 3.0, "stat": "R", "elapsed": "01:10", "command": "Codex"}},
            known_cwds=[],
        )

        self.assertTrue(cooled["active_cpu"])
        self.assertFalse(cooled["live_cpu_over_preflight_threshold"])
        self.assertEqual(cooled["cpu_blocking_threshold"], 75.0)
        self.assertTrue(hot["live_cpu_over_preflight_threshold"])

    def test_classifies_macos_spotlight_indexer_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 55,
                            "pcpu": 82.0,
                            "block_reason": "high_cpu_general",
                            "command": (
                                "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
                                "Metadata.framework/Versions/A/Support/mds_stores"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"macos_spotlight_indexer": 1})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"macos_spotlight_indexer": 1})

    def test_goal_audit_input_resolves_nested_next_mps_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested.launch_summary.json"
            audit = root / "goal_state.json"
            _write_json(
                nested,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 30,
                            "pcpu": 0.0,
                            "block_reason": "keyword:torch",
                            "command": "uv run --with torch python q.py",
                        }
                    ]
                },
            )
            _write_json(
                audit,
                {
                    "artifacts": {
                        "next_mps_quality_speed": {
                            "path": str(nested),
                        }
                    }
                },
            )

            report = diagnose.diagnose_summary(audit)

        self.assertEqual(report["summary_json"], str(nested))
        self.assertEqual(report["category_counts"], {"torch_worker": 1})

    def test_preserves_authoritative_total_when_blocker_sample_is_capped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_process_sample_limit": 32,
                    "preflight_blocking_process_count": 9,
                    "preflight_contending_process_count": 11,
                    "preflight_contending_process_sample_count": 2,
                    "preflight_blocking_processes": [
                        {"pid": 61, "pcpu": 0.0, "block_reason": "keyword:torch", "command": "torch a.py"},
                        {"pid": 62, "pcpu": 0.0, "block_reason": "keyword:torch", "command": "torch b.py"},
                    ],
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["process_sample_limit"], 32)
        self.assertEqual(report["blocker_count"], 9)
        self.assertEqual(report["blocker_sample_count"], 2)
        self.assertEqual(report["blocker_unlisted_count"], 7)
        self.assertEqual(report["contending_process_count"], 11)
        self.assertEqual(report["contending_process_sample_count"], 2)
        self.assertEqual(report["contending_process_unlisted_count"], 9)
        self.assertEqual(report["category_counts"], {"torch_worker": 2})

    def test_classifies_toto_worker_child_separately_from_generic_high_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 70,
                            "pcpu": 24.0,
                            "block_reason": "high_cpu",
                            "command": (
                                "python -m lean_trade.runners.run_btc_15m_tree_residual_live_quote_shadow_paper "
                                "--journal-path logs/btc15m_shadow_overnight/run/iterations/0001/journal.jsonl "
                                "--state-path logs/btc15m_shadow_overnight/run/live_quote_shadow_state/state.json"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"ai_trader_toto_worker": 1})
        self.assertEqual(report["live_category_counts"], {})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"ai_trader_toto_worker": 1})

    def test_classifies_btc15m_sft_worker_separately_from_toto_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 75,
                            "pcpu": 78.2,
                            "block_reason": "high_cpu",
                            "command": (
                                "python scripts/train_kalshi_btc15m_sft.py --input "
                                "/tmp/pytest-of-nicholasbardy/test_train_and_evaluate_script0/btc15m.csv"
                            ),
                        },
                        {
                            "pid": 76,
                            "pcpu": 0.0,
                            "block_reason": "periodic_mps_exporter",
                            "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                        },
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(
            report["category_counts"],
            {"ai_trader_btc15m_sft": 1, "ai_trader_toto_mps_exporter": 1},
        )

    def test_classifies_btc15m_sft_shadow_separately_from_toto_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 89,
                            "pcpu": 99.2,
                            "block_reason": "high_cpu",
                            "command": (
                                "python -m lean_trade.runners.btc_15m_sft_shadow "
                                "--config configs/lean_btc_15m_calibrated_sft_edge_paper.yaml"
                            ),
                        },
                        {
                            "pid": 90,
                            "pcpu": 0.0,
                            "block_reason": "periodic_mps_exporter",
                            "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                        },
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(
            report["category_counts"],
            {"ai_trader_btc15m_sft_shadow": 1, "ai_trader_toto_mps_exporter": 1},
        )
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"ai_trader_btc15m_sft_shadow": 1})

    def test_classifies_btc15m_sft_runtime_parity_separately_from_generic_high_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 3028,
                            "pcpu": 94.3,
                            "block_reason": "high_cpu",
                            "command": (
                                "/opt/homebrew/bin/python scripts/check_btc15m_sft_runtime_parity.py "
                                "--checkpoint /tmp/pytest-of-nicholasbardy/checkpoint "
                                "--config /tmp/pytest-of-nicholasbardy/runtime.yaml"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"ai_trader_btc15m_sft_runtime_parity": 1})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"ai_trader_btc15m_sft_runtime_parity": 1})

    def test_classifies_btc15m_activation_rl_separately_from_generic_high_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 93,
                            "pcpu": 89.5,
                            "block_reason": "high_cpu",
                            "command": (
                                "python scripts/build_btc15m_activation_rl_dataset.py "
                                "--activation-input /tmp/pytest-of-nicholasbardy/activation_bank.parquet "
                                "--output-path /tmp/pytest-of-nicholasbardy/activation_rl_dataset.parquet"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"ai_trader_btc15m_activation_rl": 1})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"ai_trader_btc15m_activation_rl": 1})

    def test_classifies_btc15m_activation_bank_integrity_separately_from_generic_high_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 1173,
                            "pcpu": 84.4,
                            "block_reason": "high_cpu",
                            "command": (
                                "/opt/homebrew/bin/python "
                                "/Users/nicholasbardy/git/ai_trader/scripts/"
                                "verify_btc15m_activation_bank_integrity.py "
                                "--activation-bank "
                                "/private/var/folders/tmp/pytest-of-nicholasbardy/"
                                "test_activation_bank_integrity1/activation_bank.parquet"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"ai_trader_btc15m_activation_bank_integrity": 1})
        self.assertEqual(
            report["active_cpu_category_counts"],
            {},
        )
        self.assertEqual(
            report["summary_cpu_active_category_counts"],
            {"ai_trader_btc15m_activation_bank_integrity": 1},
        )

    def test_classifies_btc15m_imitation_separately_from_toto_exporter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 91,
                            "pcpu": 67.2,
                            "block_reason": "high_cpu",
                            "command": (
                                "python scripts/train_kalshi_btc15m_imitation.py --input "
                                "/tmp/pytest-of-nicholasbardy/test_btc15m_imitation_tree_sou0/raw_replay.parquet"
                            ),
                        },
                        {
                            "pid": 92,
                            "pcpu": 0.0,
                            "block_reason": "periodic_mps_exporter",
                            "command": "python scripts/run_btc15m_overnight_shadow_monitor.py --run-id toto",
                        },
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(
            report["category_counts"],
            {"ai_trader_btc15m_imitation": 1, "ai_trader_toto_mps_exporter": 1},
        )
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"ai_trader_btc15m_imitation": 1})

    def test_classifies_btc15m_dqn_separately_from_generic_high_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 3417,
                            "pcpu": 99.6,
                            "block_reason": "high_cpu",
                            "command": (
                                "python scripts/train_kalshi_btc15m_dqn.py "
                                "--input /tmp/pytest-of-nicholasbardy/raw_replay.parquet "
                                "--sft-dir /tmp/pytest-of-nicholasbardy/missing_sft"
                            ),
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"ai_trader_btc15m_dqn": 1})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"ai_trader_btc15m_dqn": 1})

    def test_classifies_font_maker_standard_glyph_monitor_separately_from_generic_high_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 3421,
                            "pcpu": 9.1,
                            "block_reason": "high_cpu",
                            "command": "uv run python scripts/utilities/monitor_standard_glyph_exposure.py --notify",
                        },
                        {
                            "pid": 3422,
                            "pcpu": 8.8,
                            "block_reason": "high_cpu",
                            "command": "python scripts/utilities/monitor_standard_glyph_exposure.py --notify",
                        },
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary)

        self.assertEqual(report["category_counts"], {"font_maker_standard_glyph_monitor": 2})
        self.assertEqual(report["active_cpu_category_counts"], {})
        self.assertEqual(report["summary_cpu_active_category_counts"], {"font_maker_standard_glyph_monitor": 2})

    def test_relative_output_dir_falls_back_to_known_cwd_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ai_trader = root / "ai_trader"
            output_dir = ai_trader / "logs" / "run"
            recent = output_dir / "events.jsonl"
            recent.parent.mkdir(parents=True)
            recent.write_text("{}\n", encoding="utf-8")
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 80,
                            "pcpu": 0.0,
                            "block_reason": "periodic_mps_exporter",
                            "command": f"cd {ai_trader} && python scripts/run_btc15m_overnight_shadow_monitor.py",
                        },
                        {
                            "pid": 81,
                            "ppid": 999,
                            "pcpu": 0.0,
                            "block_reason": "periodic_mps_exporter",
                            "command": (
                                "python scripts/run_btc15m_overnight_shadow_monitor.py "
                                "--output-dir logs/run"
                            ),
                        },
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary, recent_seconds=60)

        child = report["blockers"][1]
        self.assertEqual(child["cwd"], None)
        self.assertEqual(child["output_dir"], str(output_dir))
        self.assertEqual(child["recent_outputs"][0]["path"], "events.jsonl")

    def test_ignores_old_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_dir = root / "out"
            output_dir.mkdir()
            old = output_dir / "old.jsonl"
            old.write_text("{}\n", encoding="utf-8")
            old_time = time.time() - 3600
            os.utime(old, (old_time, old_time))
            summary = root / "summary.json"
            _write_json(
                summary,
                {
                    "preflight_blocking_processes": [
                        {
                            "pid": 40,
                            "pcpu": 0.0,
                            "block_reason": "periodic_mps_exporter",
                            "command": f"python scripts/run_btc15m_overnight_shadow_monitor.py --output-dir {output_dir}",
                        }
                    ]
                },
            )

            report = diagnose.diagnose_summary(summary, recent_seconds=60)

        self.assertEqual(report["recent_output_category_counts"], {})
        self.assertEqual(report["status"], "no_live_or_recent_blockers_found")
        self.assertEqual(report["live_blocker_count"], 0)
        self.assertEqual(report["recent_output_blocker_count"], 0)
        self.assertEqual(report["live_or_recent_blocker_count"], 0)
        self.assertEqual(report["blockers"][0]["recent_output_count"], 0)


if __name__ == "__main__":
    unittest.main()
