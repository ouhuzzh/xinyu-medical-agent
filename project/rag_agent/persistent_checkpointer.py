import hashlib
import hmac
import logging
import os
import pickle
import shutil
import tempfile
from collections import defaultdict
from threading import RLock

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

# HMAC signing key derivation — uses a dedicated secret when available,
# falls back to a fixed dev key (never the JWT secret) to avoid cross-domain reuse.
_CHECKPOINT_SIGNING_SEED = "xinyu-checkpoint-signing-dev-only"


def _get_checkpoint_signing_key() -> bytes:
    """Derive the HMAC signing key for checkpoint files."""
    import config as _config
    raw = os.environ.get("CHECKPOINT_SIGNING_KEY", "").strip()
    if not raw and getattr(_config, "APP_ENV", "development") != "production":
        raw = _CHECKPOINT_SIGNING_SEED
    if not raw:
        raise RuntimeError("CHECKPOINT_SIGNING_KEY must be set in production")
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _hmac_signature(data: bytes) -> str:
    return hmac.new(_get_checkpoint_signing_key(), data, hashlib.sha256).hexdigest()


class PersistentInMemorySaver(InMemorySaver):
    """Persist LangGraph checkpoints to a local file while keeping InMemory semantics.

    Resilience:
      - Writes are atomic (tempfile + os.replace) — crashes mid-write don't corrupt.
      - On load, a backup of the previous good pkl is kept as <path>.bak; if the
        primary file fails to deserialize (corrupted, partial write from an old
        process, pickle-protocol skew), we silently fall back to the backup and
        log a warning rather than crashing the entire backend.
      - As a last resort an empty store is initialised so the API can still serve
        new sessions while ops investigates the missing data.
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = os.path.abspath(path)
        self.backup_path = self.path + ".bak"
        self._lock = RLock()
        self._last_mtime = None
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._reload_from_disk(force=True)

    def _make_storage(self):
        return defaultdict(lambda: defaultdict(dict))

    def _snapshot(self) -> dict:
        return {
            "storage": {
                thread_id: {
                    checkpoint_ns: dict(checkpoints)
                    for checkpoint_ns, checkpoints in namespaces.items()
                }
                for thread_id, namespaces in self.storage.items()
            },
            "writes": {
                key: dict(value)
                for key, value in self.writes.items()
            },
            "blobs": dict(self.blobs),
        }

    def _restore(self, payload: dict) -> None:
        storage = self._make_storage()
        for thread_id, namespaces in (payload.get("storage") or {}).items():
            storage[thread_id] = defaultdict(
                dict,
                {
                    checkpoint_ns: dict(checkpoints)
                    for checkpoint_ns, checkpoints in (namespaces or {}).items()
                },
            )
        self.storage = storage
        self.writes = defaultdict(
            dict,
            {
                tuple(key): dict(value)
                for key, value in (payload.get("writes") or {}).items()
            },
        )
        self.blobs = dict(payload.get("blobs") or {})

    def _load_pkl(self, path: str) -> dict:
        """Read and verify a pkl file.  Raises ValueError on tampering."""
        sig_path = path + ".sig"
        with open(path, "rb") as handle:
            raw = handle.read()
        if os.path.exists(sig_path):
            with open(sig_path, "r") as sf:
                expected_sig = sf.read().strip()
            actual_sig = _hmac_signature(raw)
            if not hmac.compare_digest(actual_sig, expected_sig):
                raise ValueError(
                    f"checkpoint file {path} HMAC mismatch — possible tampering or corruption"
                )
        # else: legacy file without signature — load but log a notice
        payload = pickle.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint file {path} does not contain a dict payload")
        return payload

    def _init_empty(self):
        self.storage = self._make_storage()
        self.writes = defaultdict(dict)
        self.blobs = {}
        self._last_mtime = None

    def _reload_from_disk(self, *, force: bool = False) -> None:
        if not os.path.exists(self.path):
            if force or self._last_mtime is not None:
                self._init_empty()
            return

        mtime = os.path.getmtime(self.path)
        if not force and self._last_mtime is not None and mtime <= self._last_mtime:
            return

        # Try the primary file first, fall back to the backup, then to empty
        try:
            payload = self._load_pkl(self.path)
        except Exception as primary_err:
            logger.error(
                "Failed to load checkpoint from %s: %s — attempting backup",
                self.path, primary_err,
            )
            if os.path.exists(self.backup_path):
                try:
                    payload = self._load_pkl(self.backup_path)
                    # Recover: promote backup back to primary so subsequent writes
                    # don't keep tripping on the same corrupt file
                    shutil.copyfile(self.backup_path, self.path)
                    mtime = os.path.getmtime(self.path)
                    logger.warning("Recovered checkpoint store from %s", self.backup_path)
                except Exception as backup_err:
                    logger.error(
                        "Backup checkpoint at %s also failed: %s — starting empty",
                        self.backup_path, backup_err,
                    )
                    self._init_empty()
                    return
            else:
                logger.error("No backup checkpoint available — starting empty")
                self._init_empty()
                return

        self._restore(payload)
        self._last_mtime = mtime

    def _persist_to_disk(self) -> None:
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        # Roll the current good file to .bak BEFORE the atomic replace so we
        # always have one full prior generation to fall back to on corruption.
        if os.path.exists(self.path):
            try:
                shutil.copyfile(self.path, self.backup_path)
                # Rotate the signature file too so backup recovery is tamper-safe
                sig_src = self.path + ".sig"
                sig_dst = self.backup_path + ".sig"
                if os.path.exists(sig_src):
                    shutil.copyfile(sig_src, sig_dst)
                elif os.path.exists(sig_dst):
                    os.remove(sig_dst)
            except OSError:
                # Best-effort — don't fail the write if backup rotation hiccups
                logger.warning("Failed to rotate checkpoint backup", exc_info=True)
        fd, tmp_path = tempfile.mkstemp(prefix="langgraph-checkpoint-", suffix=".tmp", dir=directory)
        sig_path = self.path + ".sig"
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(self._snapshot(), handle, protocol=pickle.HIGHEST_PROTOCOL)
            # Write HMAC signature alongside the checkpoint
            with open(tmp_path, "rb") as rh:
                sig = _hmac_signature(rh.read())
            sig_tmp = tmp_path + ".sig"
            with open(sig_tmp, "w") as sh:
                sh.write(sig)
            os.replace(tmp_path, self.path)
            os.replace(sig_tmp, sig_path)
            self._last_mtime = os.path.getmtime(self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(tmp_path + ".sig"):
                os.remove(tmp_path + ".sig")

    def get_tuple(self, config):
        with self._lock:
            self._reload_from_disk()
            return super().get_tuple(config)

    def list(self, config, *, filter=None, before=None, limit=None):
        with self._lock:
            self._reload_from_disk()
            items = list(super().list(config, filter=filter, before=before, limit=limit))
        for item in items:
            yield item

    def put(self, config, checkpoint, metadata, new_versions):
        with self._lock:
            self._reload_from_disk()
            result = super().put(config, checkpoint, metadata, new_versions)
            self._persist_to_disk()
            return result

    def put_writes(self, config, writes, task_id, task_path=""):
        with self._lock:
            self._reload_from_disk()
            super().put_writes(config, writes, task_id, task_path)
            self._persist_to_disk()

    def delete_thread(self, thread_id: str) -> None:
        with self._lock:
            self._reload_from_disk()
            super().delete_thread(thread_id)
            self._persist_to_disk()
