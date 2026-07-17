from __future__ import annotations

import unittest
from unittest import mock

import train_eval_owner_run_tape as train_eval


class TrainEvalBenchmarkEnvironmentTests(unittest.TestCase):
    def test_wait_for_benchmark_environment_returns_when_clean(self) -> None:
        with (
            mock.patch.object(
                train_eval,
                "_capture_benchmark_environment",
                side_effect=[
                    {"status": "contended", "blocking_processes": [{"pid": 1}]},
                    {"status": "background", "blocking_processes": []},
                ],
            ),
            mock.patch.object(train_eval.time, "sleep") as sleep_mock,
        ):
            environment = train_eval._wait_for_benchmark_environment_ok(timeout_s=10.0, poll_s=0.25)

        self.assertEqual(environment["status"], "background")
        sleep_mock.assert_called_once_with(0.25)

    def test_wait_for_benchmark_environment_returns_contended_after_zero_timeout(self) -> None:
        with (
            mock.patch.object(
                train_eval,
                "_capture_benchmark_environment",
                return_value={"status": "contended", "blocking_processes": [{"pid": 1}]},
            ),
            mock.patch.object(train_eval.time, "sleep") as sleep_mock,
        ):
            environment = train_eval._wait_for_benchmark_environment_ok(timeout_s=0.0, poll_s=0.25)

        self.assertEqual(environment["status"], "contended")
        sleep_mock.assert_not_called()

    def test_wait_for_benchmark_environment_treats_unchecked_as_not_promotable(self) -> None:
        with (
            mock.patch.object(
                train_eval,
                "_capture_benchmark_environment",
                return_value={"status": "unchecked", "error": "ps failed"},
            ),
            mock.patch.object(train_eval.time, "sleep") as sleep_mock,
        ):
            environment = train_eval._wait_for_benchmark_environment_ok(timeout_s=0.0, poll_s=0.25)

        self.assertEqual(environment["status"], "unchecked")
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
