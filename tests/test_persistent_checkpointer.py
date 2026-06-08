import os
import shutil
import sys
import tempfile
import unittest
from typing import TypedDict

sys.path.insert(0, r"D:\nageoffer\agentic-rag-for-dummies\project")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from rag_agent.persistent_checkpointer import PersistentInMemorySaver  # noqa: E402


class CounterState(TypedDict):
    value: int


def _build_graph(saver):
    builder = StateGraph(CounterState)
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    return builder.compile(checkpointer=saver)


class PersistentCheckpointerTests(unittest.TestCase):
    def test_checkpoint_survives_reopen_and_delete_thread_clears_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "langgraph-checkpoints.pkl")
            saver = PersistentInMemorySaver(path)
            graph = _build_graph(saver)

            config = {"configurable": {"thread_id": "thread-persist"}}
            result = graph.invoke({"value": 1}, config=config)

            self.assertEqual(result["value"], 2)
            reopened = PersistentInMemorySaver(path)
            self.assertIsNotNone(reopened.get_tuple(config))

            reopened.delete_thread("thread-persist")
            again = PersistentInMemorySaver(path)
            self.assertIsNone(again.get_tuple(config))

    def test_backup_is_rotated_on_every_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pkl")
            saver = PersistentInMemorySaver(path)
            graph = _build_graph(saver)

            # First write — primary exists, backup does not yet (nothing to rotate)
            graph.invoke({"value": 1}, config={"configurable": {"thread_id": "t1"}})
            self.assertTrue(os.path.exists(path))
            # Second write — backup should now hold the snapshot from the first write
            graph.invoke({"value": 1}, config={"configurable": {"thread_id": "t2"}})
            self.assertTrue(os.path.exists(path + ".bak"))

    def test_corrupt_primary_recovers_from_backup(self):
        """A truncated/garbage primary pkl must fall back to the .bak file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pkl")
            saver = PersistentInMemorySaver(path)
            graph = _build_graph(saver)

            # Generate two writes so backup is populated with valid data
            cfg1 = {"configurable": {"thread_id": "thread-a"}}
            cfg2 = {"configurable": {"thread_id": "thread-b"}}
            graph.invoke({"value": 1}, config=cfg1)
            graph.invoke({"value": 1}, config=cfg2)

            # Corrupt the primary
            with open(path, "wb") as fh:
                fh.write(b"\x00\xFF\x00 not a valid pickle stream")

            # Re-open — should recover from backup (thread-a survives, thread-b too
            # since the backup snapshot was taken AFTER the second put rotated it)
            recovered = PersistentInMemorySaver(path)
            # At minimum thread-a should be retrievable
            self.assertIsNotNone(recovered.get_tuple(cfg1))

    def test_missing_primary_and_backup_starts_empty(self):
        """No files at all — should NOT raise; serve from empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "missing.pkl")
            saver = PersistentInMemorySaver(path)
            # Empty store — any get returns None
            cfg = {"configurable": {"thread_id": "ghost"}}
            self.assertIsNone(saver.get_tuple(cfg))

    def test_corrupt_primary_and_backup_starts_empty(self):
        """Both files corrupt — should NOT crash on init; degrade to empty store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pkl")
            # Write garbage to both
            with open(path, "wb") as fh:
                fh.write(b"junk")
            with open(path + ".bak", "wb") as fh:
                fh.write(b"also junk")

            # Must not raise
            saver = PersistentInMemorySaver(path)
            cfg = {"configurable": {"thread_id": "nothing"}}
            self.assertIsNone(saver.get_tuple(cfg))

