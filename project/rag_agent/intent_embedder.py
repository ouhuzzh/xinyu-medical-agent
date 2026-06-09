"""L2 Intent Embedding Layer — semantic intent matching via bge-m3.

Uses the project's existing embedding model to compute utterance→intent similarity.
Utterances are loaded from the skill registry at init time — each skill declares
its own utterances via BaseSkill.utterances.

Design:
- Singleton, lazy-init on first classify() call
- Pre-computes intent centroids from registry-loaded utterances
- Cosine similarity > threshold → (intent, confidence); else → None → L3 LLM

Threshold is configurable via INTENT_EMBEDDING_THRESHOLD env var (default 0.50).
"""

from __future__ import annotations

import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback utterances — used when skill registry is not available.
# In production, utterances are loaded from the skill registry at init time.
# ---------------------------------------------------------------------------

_FALLBACK_UTTERANCES: Dict[str, List[str]] = {
    "greeting": ["你好", "谢谢", "再见", "hello", "hi", "bye"],
    "appointment": ["我要挂号", "帮我预约", "我想预约看病"],
    "cancel_appointment": ["取消预约", "退号", "取消刚才的挂号"],
    "triage": ["挂什么科", "看哪个科", "应该看什么科室"],
    "medical_rag": [
        "预约前要注意什么", "感冒吃什么药", "高血压怎么控制",
        "头痛是什么原因", "严重吗", "怎么办",
    ],
}


def _load_utterances() -> Dict[str, List[str]]:
    """Load utterances from skill registry, or fall back to hardcoded defaults."""
    try:
        from skills.registry import get_skill_registry
        registry = get_skill_registry()
        if registry.skills:
            utterances = registry.collect_utterances()
            if utterances:
                return utterances
    except Exception:
        pass
    return dict(_FALLBACK_UTTERANCES)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalised or raw vectors."""
    a, b = np.asarray(a), np.asarray(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class IntentEmbedder:
    """Semantic intent matcher using the project's embedding model.

    Usage::

        embedder = get_intent_embedder()
        result = embedder.classify("我要挂号看心内科")
        # → ("appointment", 0.87)  or  None if below threshold
    """

    def __init__(self, threshold: float | None = None):
        self._threshold = threshold  # None = use config default
        self._centroids: Dict[str, np.ndarray] = {}
        self._embedding_model = None
        self._initialized = False
        self._init_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> Optional[Tuple[str, float]]:
        """Return (intent_label, confidence) or None if below threshold.

        Confidence is the cosine similarity between the query embedding and
        the closest intent centroid.  Returns None when no intent exceeds
        the threshold — callers should fall through to L3 LLM.
        """
        self._lazy_init()
        if not self._centroids:
            return None

        query_vec = self._encode(query)
        best_intent: Optional[str] = None
        best_score: float = -1.0

        for intent, centroid in self._centroids.items():
            score = _cosine_similarity(query_vec, centroid)
            if score > best_score:
                best_score = score
                best_intent = intent

        threshold = self._threshold if self._threshold is not None else _default_threshold()
        if best_intent and best_score >= threshold:
            return (best_intent, float(best_score))
        return None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _lazy_init(self):
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:  # double-check after acquiring lock
                return
            try:
                embedding_model = self._load_embedding_model()
                centroids = self._compute_centroids(embedding_model)
                # Assign atomically from locals — no thread sees partial state
                self._embedding_model = embedding_model
                self._centroids = centroids
                self._initialized = True
                threshold = self._threshold if self._threshold is not None else _default_threshold()
                logger.info(
                    "IntentEmbedder ready: %d intents, threshold=%.2f",
                    len(centroids), threshold,
                )
            except Exception:
                logger.exception("IntentEmbedder init failed — L2 embedding layer disabled.")
                self._centroids = {}
                self._initialized = True  # don't retry every request

    def _load_embedding_model(self):
        """Return the project's shared embedding model (LangChain Embeddings)."""
        from model_factory import get_embedding_model
        return get_embedding_model()

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text into a numpy vector."""
        vec = self._embedding_model.embed_query(text)
        return np.asarray(vec, dtype=np.float32)

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts in one batch call."""
        vecs = self._embedding_model.embed_documents(texts)
        return [np.asarray(v, dtype=np.float32) for v in vecs]

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    def _compute_centroids(self, embedding_model) -> Dict[str, np.ndarray]:
        """Pre-compute intent centroids as the mean of utterance embeddings."""
        centroids: Dict[str, np.ndarray] = {}
        for intent_label, utterances in _load_utterances().items():
            if not utterances:
                continue
            vecs = self._encode_batch(utterances)
            centroid = np.mean(np.stack(vecs), axis=0)
            # Normalise for cosine-similarity convenience
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[intent_label] = centroid
        return centroids


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _default_threshold() -> float:
    """Read INTENT_EMBEDDING_THRESHOLD from config, default 0.70."""
    try:
        import config
        return float(getattr(config, "INTENT_EMBEDDING_THRESHOLD", 0.70) or 0.70)
    except Exception:
        return 0.70


# ---------------------------------------------------------------------------
# Singleton access
# ---------------------------------------------------------------------------

_intent_embedder: Optional[IntentEmbedder] = None


def get_intent_embedder() -> IntentEmbedder:
    """Return the singleton IntentEmbedder instance."""
    global _intent_embedder
    if _intent_embedder is None:
        _intent_embedder = IntentEmbedder()
    return _intent_embedder
