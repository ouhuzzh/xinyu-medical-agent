import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project"))

from core.knowledge_base_worker import KnowledgeBaseWorker  # noqa: E402


class FakeRunner:
    def __init__(self, *, fail_first_sync=False):
        self.bootstrap_calls = 0
        self.sync_calls = 0
        self.close_calls = 0
        self.fail_first_sync = fail_first_sync

    def bootstrap(self):
        self.bootstrap_calls += 1
        return {"status": "ready"}

    def sync_all(self):
        self.sync_calls += 1
        if self.fail_first_sync and self.sync_calls == 1:
            raise RuntimeError("temporary failure")
        return []

    def close(self):
        self.close_calls += 1


class FakeStopEvent:
    def __init__(self, waits_before_stop):
        self.waits_before_stop = waits_before_stop
        self.wait_calls = []

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        self.waits_before_stop -= 1
        return self.waits_before_stop < 0


class KnowledgeBaseWorkerTests(unittest.TestCase):
    def test_runs_bootstrap_then_scheduled_sync_and_closes(self):
        runner = FakeRunner()
        stop_event = FakeStopEvent(waits_before_stop=1)
        worker = KnowledgeBaseWorker(
            runner,
            bootstrap_on_start=True,
            sync_enabled=True,
            sync_interval_seconds=15,
        )

        worker.run_forever(stop_event)

        self.assertEqual(runner.bootstrap_calls, 1)
        self.assertEqual(runner.sync_calls, 1)
        self.assertEqual(runner.close_calls, 1)
        self.assertEqual(stop_event.wait_calls, [15, 15])

    def test_sync_failure_does_not_stop_future_runs(self):
        runner = FakeRunner(fail_first_sync=True)
        stop_event = FakeStopEvent(waits_before_stop=2)
        worker = KnowledgeBaseWorker(
            runner,
            bootstrap_on_start=False,
            sync_enabled=True,
            sync_interval_seconds=5,
        )

        worker.run_forever(stop_event)

        self.assertEqual(runner.sync_calls, 2)
        self.assertEqual(runner.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
