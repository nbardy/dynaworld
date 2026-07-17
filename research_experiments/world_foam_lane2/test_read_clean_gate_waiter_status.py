from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import read_clean_gate_waiter_status as reader


class ReadCleanGateWaiterStatusTests(unittest.TestCase):
    def test_newest_waiter_summary_prefers_latest_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "older_clean_mps_wait.json"
            newer = root / "newer_clean_mps_wait.json"
            ignored = root / "other.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            ignored.write_text("{}", encoding="utf-8")
            os.utime(older, (100.0, 100.0))
            os.utime(newer, (200.0, 200.0))
            os.utime(ignored, (300.0, 300.0))
            self.assertEqual(reader._newest_waiter_summary(root), newer)

    def test_sample_processes_keeps_compact_process_fields(self) -> None:
        payload = {
            "current_benchmark_environment_blocking_process_sample": [
                {
                    "pid": 1,
                    "ppid": 0,
                    "pcpu": 12.5,
                    "pmem": 3.0,
                    "elapsed": "00:01",
                    "block_reason": "high_cpu",
                    "screen_session_name": None,
                    "command": "python train.py",
                    "extra": "ignored",
                },
                {"pid": 2, "command": "python other.py"},
            ]
        }
        self.assertEqual(
            reader._sample_processes(payload, limit=1),
            [
                {
                    "pid": 1,
                    "ppid": 0,
                    "pcpu": 12.5,
                    "pmem": 3.0,
                    "elapsed": "00:01",
                    "block_reason": "high_cpu",
                    "screen_session_name": None,
                    "command": "python train.py",
                }
            ],
        )

    def test_sample_processes_tolerates_missing_sample(self) -> None:
        self.assertEqual(reader._sample_processes({}, limit=5), [])

    def test_summary_stale_for_poll_uses_double_poll_window(self) -> None:
        self.assertFalse(reader._is_stale_for_poll({"wait_ready_poll_s": 30.0}, age_s=60.0))
        self.assertTrue(reader._is_stale_for_poll({"wait_ready_poll_s": 30.0}, age_s=60.1))
        self.assertIsNone(reader._is_stale_for_poll({"wait_ready_poll_s": 0.0}, age_s=100.0))

    def test_summary_age_is_nonnegative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample_clean_mps_wait.json"
            path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            os.utime(path, (10.0, 10.0))
            self.assertEqual(reader._summary_age_s(path, now_ns=5_000_000_000), 0.0)
            self.assertEqual(reader._summary_age_s(path, now_ns=12_500_000_000), 2.5)

    def test_wait_for_newer_summary_returns_current_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "sample_clean_mps_wait.json"
            path.write_text("{}", encoding="utf-8")
            os.utime(path, (100.0, 100.0))
            self.assertEqual(
                reader._wait_for_newer_summary(
                    results_dir=root,
                    initial_mtime_ns=path.stat().st_mtime_ns,
                    timeout_s=0.0,
                    poll_s=0.05,
                ),
                path,
            )

    def test_wait_for_newer_summary_prefers_new_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "older_clean_mps_wait.json"
            newer = root / "newer_clean_mps_wait.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            os.utime(older, (100.0, 100.0))
            os.utime(newer, (200.0, 200.0))
            self.assertEqual(
                reader._wait_for_newer_summary(
                    results_dir=root,
                    initial_mtime_ns=older.stat().st_mtime_ns,
                    timeout_s=0.0,
                    poll_s=0.05,
                ),
                newer,
            )


if __name__ == "__main__":
    unittest.main()
