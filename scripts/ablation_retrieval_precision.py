"""Ablation study for retrieval Precision@5 uplift (Resume bullet #3).

Self-contained — no heavy project imports. Copies the minimal classes
needed from project/benchmarks/resume_benchmarks.py and
project/rag_agent/tools.py.

Ablation variants (incrementally adding components)
---------------------------------------------------
1. BASELINE          – vector search only, flat chunks
2. +HYBRID          – add keyword search + RRF fusion
3. +QUERY_PLAN      – add multi-query planning (plan_queries)
4. +PARENT_CHILD    – use parent/child chunk index
5. +RERANK          – add reranking
6. +SOURCE_PRIORITY – add source-type priority sorting
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_PATH = REPO_ROOT / "scripts" / "ablation_data" / "retrieval_samples_extended.json"
DOC_PATTERNS = ("who-*.md", "nhc-*.md")
TOP_K = 5
MRR_K = 10

# ── Data classes ──────────────────────────────────────────────────

@dataclass
class MedicalRagBenchmarkSample:
    id: str
    question: str
    expected_sources: list[str]
    expected_keywords: list[str] = field(default_factory=list)
    search_query: str = ""
    source_types: list[str] = field(default_factory=list)
    topic_focus: str = ""
    recent_context: str = ""
    category: str = ""


# ── Embeddings ────────────────────────────────────────────────────

class KeywordBenchmarkEmbeddings:
    CONCEPT_ALIASES = (
        ("hypertension", ("高血压", "hypertension", "blood pressure")),
        ("hypertension_lifestyle", ("低盐", "限盐", "high-salt diet", "physical activity", "alcohol", "tobacco", "lifestyle")),
        ("hypertension_symptoms", ("头晕", "headaches", "chest pain", "blood pressure checked", "checked")),
        ("diabetes", ("糖尿病", "diabetes", "blood glucose", "type 2")),
        ("diabetes_prevention", ("healthy diet", "physical activity", "body weight", "tobacco", "体重")),
        ("diabetes_complications", ("blindness", "kidney failure", "heart attacks", "stroke", "失明", "肾衰竭")),
        ("asthma", ("哮喘", "asthma", "wheezing", "shortness of breath")),
        ("asthma_management", ("inhaler", "bronchodilators", "steroids", "triggers", "诱因", "吸入")),
        ("covid", ("新冠", "covid", "sars-cov-2")),
        ("covid_transmission", ("飞沫", "密切接触", "气溶胶", "污染的物品")),
        ("covid_prevention", ("疫苗", "加强免疫", "勤洗手", "戴口罩", "通风")),
        ("covid_risk", ("老年人", "严重基础疾病", "晚期妊娠", "肥胖", "重症")),
        ("influenza", ("流感", "influenza")),
        ("influenza_transmission", ("空气传播", "人群密集", "通风不良", "咳嗽")),
        ("mycoplasma", ("肺炎支原体", "mycoplasma", "儿童")),
        ("mycoplasma_severe", ("发热", "咳嗽", "呼吸困难", "胸痛")),
    )

    def _vector(self, text: str) -> list[float]:
        normalized = str(text or "").lower()
        return [float(sum(normalized.count(alias.lower()) for alias in aliases)) for _, aliases in self.CONCEPT_ALIASES]

    def embed_documents(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


# ── In-memory collection ──────────────────────────────────────────

class InMemoryCollection:
    def __init__(self, docs: list[Document], embeddings: KeywordBenchmarkEmbeddings):
        self.docs = [self._clone(doc) for doc in docs]
        self.embeddings = embeddings
        self.doc_vectors = embeddings.embed_documents(self._doc_text(doc) for doc in self.docs)

    @staticmethod
    def _clone(doc):
        return Document(page_content=doc.page_content, metadata=dict(doc.metadata or {}))

    @staticmethod
    def _doc_text(doc):
        m = doc.metadata or {}
        return " ".join(str(p).strip() for p in (m.get("title"), m.get("document_topic"), m.get("section_title"), doc.page_content) if str(p or "").strip())

    @staticmethod
    def _source_type(doc):
        m = doc.metadata or {}
        return str(m.get("source_type") or m.get("document_type") or m.get("intended_audience") or "").strip().lower()

    @staticmethod
    def _cosine(left, right):
        num = sum(a * b for a, b in zip(left, right))
        ln = math.sqrt(sum(a * a for a in left))
        rn = math.sqrt(sum(b * b for b in right))
        return num / (ln * rn) if ln and rn else 0.0

    @staticmethod
    def _normalize_source_types(st):
        return {str(s).strip().lower() for s in (st or []) if str(s).strip()}

    def _extract_terms(self, query):
        n = query.lower()
        terms = {t for t in re.findall(r"[a-z0-9]+|[一-鿿]{2,}", n) if len(t) >= 2}
        for _, aliases in KeywordBenchmarkEmbeddings.CONCEPT_ALIASES:
            for a in aliases:
                if a.lower() in n:
                    terms.add(a.lower())
        return sorted(terms)

    def _lexical_score(self, query, doc):
        haystack = self._doc_text(doc).lower()
        terms = self._extract_terms(query)
        return float(sum(haystack.count(t) for t in terms)) if terms else 0.0

    def similarity_search(self, query, k=4, score_threshold=0.0, source_types=None, rerank=True):
        allowed = self._normalize_source_types(source_types)
        qv = self.embeddings.embed_query(query)
        scored = []
        for doc, vec in zip(self.docs, self.doc_vectors):
            if allowed and self._source_type(doc) not in allowed:
                continue
            s = self._cosine(qv, vec)
            if s < score_threshold:
                continue
            c = self._clone(doc)
            c.metadata["score"] = round(s, 6)
            scored.append(c)
        scored.sort(key=lambda d: float((d.metadata or {}).get("score") or 0), reverse=True)
        results = scored[:k]
        if rerank:
            return self.rerank_candidates(query, results, k)
        return results

    def keyword_search(self, query, k=4, source_types=None):
        allowed = self._normalize_source_types(source_types)
        scored = []
        for doc in self.docs:
            if allowed and self._source_type(doc) not in allowed:
                continue
            s = self._lexical_score(query, doc)
            if s <= 0:
                continue
            c = self._clone(doc)
            c.metadata["score"] = round(s, 6)
            scored.append(c)
        scored.sort(key=lambda d: float((d.metadata or {}).get("score") or 0), reverse=True)
        return scored[:k]

    def rerank_candidates(self, query, candidates, top_n):
        qv = self.embeddings.embed_query(query)
        reranked = []
        for c in candidates:
            sem = self._cosine(qv, self.embeddings.embed_query(self._doc_text(c)))
            lex = self._lexical_score(query, c)
            fused = (sem * 0.7) + (lex * 0.3)
            r = self._clone(c)
            r.metadata["score"] = round(fused, 6)
            reranked.append(r)
        reranked.sort(key=lambda d: float((d.metadata or {}).get("score") or 0), reverse=True)
        return reranked[:top_n]

    def filtered(self, source_types=None):
        allowed = self._normalize_source_types(source_types)
        if not allowed:
            return self
        return InMemoryCollection([self._clone(d) for d in self.docs if self._source_type(d) in allowed], self.embeddings)


# ── RRF fusion ────────────────────────────────────────────────────

_RRF_K = 60

def _doc_key(doc):
    m = doc.metadata or {}
    return (m.get("parent_id") or m.get("baseline_chunk_id") or m.get("chunk_id") or "",
            m.get("source") or "", doc.page_content[:80])

def _doc_score(metadata):
    for key in ("rerank_score", "score", "fusion_score"):
        v = metadata.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0

def rrf_fuse_ranked_sets(ranked_sets, limit):
    fused_scores = {}
    chosen = {}
    best_raw = {}
    for rs in ranked_sets:
        for rank, doc in enumerate(rs, start=1):
            key = _doc_key(doc)
            fused_scores[key] = fused_scores.get(key, 0) + (1.0 / (_RRF_K + rank))
            raw = _doc_score(doc.metadata)
            if key not in chosen or raw >= best_raw.get(key, 0):
                chosen[key] = doc
                best_raw[key] = raw
    result = []
    for key, doc in chosen.items():
        c = InMemoryCollection._clone(doc)
        c.metadata["fusion_score"] = round(fused_scores[key], 6)
        if not c.metadata.get("score"):
            c.metadata["score"] = c.metadata["fusion_score"]
        result.append(c)
    result.sort(key=lambda d: float((d.metadata or {}).get("fusion_score") or 0), reverse=True)
    return result[:limit]

def rrf_fuse_two(vector_results, keyword_results, limit):
    return rrf_fuse_ranked_sets([vector_results, keyword_results], limit)


# ── Query planning ────────────────────────────────────────────────

_QUERY_TYPE_KEYWORDS = {
    "clinical_guideline": ("指南", "诊疗", "方案", "guideline", "protocol"),
    "public_health": ("预防", "传播", "风险", "prevention", "transmission", "risk"),
}

def plan_queries(query, topic_focus="", recent_context=""):
    base = str(query or "").strip()
    if not base:
        return []
    planned = [base]
    normalized = base.lower()
    if topic_focus and topic_focus.strip() and topic_focus.strip() not in base:
        planned.append(f"{topic_focus.strip()} {base}".strip())
    if recent_context.strip() and any(t in normalized for t in ("那", "这个", "这种情况", "这会", "还要", "要紧吗")):
        planned.append(f"{recent_context.strip()} {base}".strip())
    if any(k in normalized for k in _QUERY_TYPE_KEYWORDS["clinical_guideline"]):
        planned.append(f"{base} 指南 诊疗方案")
    elif any(k in normalized for k in _QUERY_TYPE_KEYWORDS["public_health"]):
        planned.append(f"{base} 预防 风险 传播")
    else:
        planned.append(f"{base} 症状 治疗 注意事项")
    seen = set()
    deduped = []
    for q in planned:
        key = re.sub(r"\s+", " ", q).strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped[:4]


# ── Source priority ───────────────────────────────────────────────

_SOURCE_PRIORITY = {"patient_education": 0, "public_health": 1, "clinical_guideline": 2}

def preferred_source_layers(query):
    n = (query or "").lower()
    if any(k in n for k in _QUERY_TYPE_KEYWORDS["clinical_guideline"]):
        return ["clinical_guideline", "public_health", "patient_education"]
    if any(k in n for k in _QUERY_TYPE_KEYWORDS["public_health"]):
        return ["public_health", "patient_education", "clinical_guideline"]
    return ["patient_education", "public_health", "clinical_guideline"]

def sort_by_source_priority(docs, preferred_layers):
    def _priority(doc):
        st = (doc.metadata or {}).get("source_type", "").lower()
        try:
            return preferred_layers.index(st)
        except ValueError:
            return 99
    return sorted(docs, key=_priority)


# ── Document corpora ──────────────────────────────────────────────

def _extract_front_matter(raw):
    metadata = {}
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            if metadata:
                break
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", s)
        if not m:
            if metadata:
                break
            continue
        metadata[m.group(1).strip().lower().replace(" ", "_")] = m.group(2).strip()
    return metadata

def _strip_front_matter(raw):
    lines = raw.splitlines()
    out, seen_meta, meta_done = [], False, False
    for line in lines:
        s = line.strip()
        is_meta = bool(re.match(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*(.+?)\s*$", s))
        if not meta_done and is_meta:
            seen_meta = True
            continue
        if seen_meta and not meta_done and not s:
            meta_done = True
            continue
        if meta_done or not seen_meta:
            out.append(line)
    return "\n".join(out).strip()

def build_corpora(doc_paths):
    """Build baseline (flat chunks) and optimized (parent/child chunks) corpora."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=180)
    baseline_docs = []
    optimized_docs = []
    for dp in doc_paths:
        raw = dp.read_text(encoding="utf-8")
        fm = _extract_front_matter(raw)
        content = _strip_front_matter(raw)
        source_type = fm.get("document_type") or fm.get("source_type") or "general"
        title = fm.get("title") or dp.stem
        # Baseline: flat chunks
        for i, chunk in enumerate(splitter.split_text(content)):
            baseline_docs.append(Document(page_content=chunk, metadata={
                "source": dp.name, "source_type": source_type, "title": title,
                "baseline_chunk_id": f"{dp.stem}_baseline_{i}",
            }))
        # Optimized: parent/child via markdown header splits
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "h1"), ("##", "h2"), ("###", "h3"),
        ])
        try:
            parent_chunks = md_splitter.split_text(content)
        except Exception:
            parent_chunks = [Document(page_content=content, metadata={})]
        # Merge small parents
        merged = []
        for pc in parent_chunks:
            pc.metadata.update({"source": dp.name, "source_type": source_type, "title": title, "document_topic": title})
            if merged and len(pc.page_content) < 200:
                merged[-1] = Document(page_content=merged[-1].page_content + "\n\n" + pc.page_content,
                                       metadata={**merged[-1].metadata, **pc.metadata})
            else:
                merged.append(pc)
        # Split large parents into children
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
        pid = 0
        for parent in merged:
            pid += 1
            parent_id = f"{dp.stem}_parent_{pid}"
            parent.metadata["parent_id"] = parent_id
            children = child_splitter.split_documents([parent])
            for ci, child in enumerate(children):
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_id"] = f"{parent_id}_child_{ci}"
                child.metadata["source"] = child.metadata.get("source") or dp.name
                child.metadata["source_type"] = child.metadata.get("source_type") or source_type
                child.metadata["title"] = child.metadata.get("title") or title
                child.metadata["document_topic"] = child.metadata.get("document_topic") or title
            optimized_docs.extend(children)
    return baseline_docs, optimized_docs


# ── Metrics ───────────────────────────────────────────────────────

def _unique_sources(docs, limit):
    ordered, seen = [], set()
    for doc in docs:
        s = str((doc.metadata or {}).get("source") or "").strip()
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
            if len(ordered) >= limit:
                break
    return ordered

def source_metrics(expected_sources, docs, top_k, mrr_k):
    norm = {s.strip().lower() for s in expected_sources if s.strip()}
    top = _unique_sources(docs, max(top_k, mrr_k))
    top_k_sources = top[:top_k]
    hits = sum(1 for s in top_k_sources if s.lower() in norm)
    rr = 0.0
    for rank, s in enumerate(top[:mrr_k], start=1):
        if s.lower() in norm:
            rr = round(1.0 / rank, 4)
            break
    return {
        "precision_at_k": round(hits / len(top_k_sources), 4) if top_k_sources else 0,
        "recall_at_k": round(hits / len(norm), 4) if norm else 0,
        "mrr_at_k": rr,
        "hit_at_k": 1.0 if hits else 0.0,
    }

def keyword_coverage(expected_keywords, docs):
    kws = [k.strip().lower() for k in expected_keywords if k.strip()]
    if not kws:
        return 0.0
    haystack = "\n".join(d.page_content for d in docs).lower()
    return round(sum(1 for k in kws if k in haystack) / len(kws), 4)


# ── Main ablation ─────────────────────────────────────────────────

def load_samples():
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return [MedicalRagBenchmarkSample(
        id=item["id"], question=item["question"],
        expected_sources=item.get("expected_sources", []),
        expected_keywords=item.get("expected_keywords", []),
        search_query=item.get("search_query", ""),
        source_types=item.get("source_types", []),
        topic_focus=item.get("topic_focus", ""),
        recent_context=item.get("recent_context", ""),
        category=item.get("category", ""),
    ) for item in payload]


def _run_variant(sample, coll, *, use_hybrid, use_query_plan, use_rerank, use_source_priority):
    query = sample.search_query or sample.question
    active = coll.filtered(sample.source_types) if sample.source_types else coll

    queries = plan_queries(query, topic_focus=sample.topic_focus, recent_context=sample.recent_context) if use_query_plan else [query]

    ranked_sets = []
    for q in queries:
        vec = active.similarity_search(q, k=max(TOP_K * 2, MRR_K), score_threshold=0.0, rerank=False)
        if use_hybrid:
            kw = active.keyword_search(q, k=max(TOP_K * 2, MRR_K), source_types=sample.source_types)
            fused = rrf_fuse_two(vec, kw, max(TOP_K * 2, MRR_K))
        else:
            fused = vec
        ranked_sets.append(fused)

    if len(ranked_sets) > 1:
        docs = rrf_fuse_ranked_sets(ranked_sets, max(TOP_K, MRR_K))
    else:
        docs = ranked_sets[0] if ranked_sets else []

    if use_source_priority:
        docs = sort_by_source_priority(docs, preferred_source_layers(query))

    if use_rerank:
        docs = active.rerank_candidates(query, docs, max(TOP_K, MRR_K))

    docs = docs[:max(TOP_K, MRR_K)]
    m = source_metrics(sample.expected_sources, docs, TOP_K, MRR_K)
    return {**m, "keyword_coverage": keyword_coverage(sample.expected_keywords, docs[:TOP_K])}


def run_ablation():
    doc_paths = []
    markdown_dir = REPO_ROOT / "markdown_docs"
    for pattern in DOC_PATTERNS:
        doc_paths.extend(sorted(markdown_dir.glob(pattern)))
    if not doc_paths:
        print("ERROR: No docs under markdown_docs/")
        return

    samples = load_samples()
    baseline_docs, optimized_docs = build_corpora(doc_paths)
    embeddings = KeywordBenchmarkEmbeddings()
    baseline_coll = InMemoryCollection(baseline_docs, embeddings)
    optimized_coll = InMemoryCollection(optimized_docs, embeddings)

    VARIANTS = [
        ("1_BASELINE",         dict(coll="baseline", use_hybrid=False, use_query_plan=False, use_rerank=False, use_source_priority=False)),
        ("2_+HYBRID",          dict(coll="baseline", use_hybrid=True,  use_query_plan=False, use_rerank=False, use_source_priority=False)),
        ("3_+QUERY_PLAN",      dict(coll="baseline", use_hybrid=True,  use_query_plan=True,  use_rerank=False, use_source_priority=False)),
        ("4_+PARENT_CHILD",    dict(coll="optimized", use_hybrid=True, use_query_plan=True,  use_rerank=False, use_source_priority=False)),
        ("5_+RERANK",          dict(coll="optimized", use_hybrid=True, use_query_plan=True,  use_rerank=True,  use_source_priority=False)),
        ("6_+SOURCE_PRIORITY", dict(coll="optimized", use_hybrid=True, use_query_plan=True,  use_rerank=True,  use_source_priority=True)),
    ]

    all_rows = {v[0]: [] for v in VARIANTS}

    for sample in samples:
        for vname, vcfg in VARIANTS:
            coll = baseline_coll if vcfg["coll"] == "baseline" else optimized_coll
            result = _run_variant(sample, coll,
                use_hybrid=vcfg["use_hybrid"],
                use_query_plan=vcfg["use_query_plan"],
                use_rerank=vcfg["use_rerank"],
                use_source_priority=vcfg["use_source_priority"])
            result["sample_id"] = sample.id
            all_rows[vname].append(result)

    def _avg(rows, key):
        vals = [r[key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    variant_summaries = []
    for vname, _ in VARIANTS:
        rows = all_rows[vname]
        s = {k: _avg(rows, k) for k in ("precision_at_k", "recall_at_k", "mrr_at_k", "hit_at_k", "keyword_coverage")}
        s["variant"] = vname
        variant_summaries.append(s)

    for i in range(1, len(variant_summaries)):
        prev = variant_summaries[i - 1]
        curr = variant_summaries[i]
        curr["precision_delta"] = round(curr["precision_at_k"] - prev["precision_at_k"], 4)

    # ── Print report ──
    print("=" * 80)
    print("ABLATION STUDY: Retrieval Precision@5 Uplift (Extended)")
    print("=" * 80)
    print(f"\nSamples: {len(samples)}   Corpus: NHC/WHO/medlineplus   top_k={TOP_K}")
    print(f"Baseline: vector-only on flat chunks (RecursiveCharacterTextSplitter 1200/180)\n")

    print(f"{'Variant':<25} {'P@5':>6} {'R@5':>6} {'MRR@10':>7} {'Hit@5':>6} {'KW_Cov':>7} {'ΔP@5':>7}")
    print("-" * 70)
    for s in variant_summaries:
        d = s.get("precision_delta", "")
        ds = f"{d:+.4f}" if isinstance(d, float) else ""
        print(f"{s['variant']:<25} {s['precision_at_k']:>6.4f} {s['recall_at_k']:>6.4f} "
              f"{s['mrr_at_k']:>7.4f} {s['hit_at_k']:>6.4f} {s['keyword_coverage']:>7.4f} {ds:>7}")

    # ── Group by query category ──
    print("\n" + "=" * 80)
    print("P@5 BY QUERY CATEGORY")
    print("=" * 80)
    categories = sorted(set(s.category for s in samples))
    for cat in categories:
        cat_samples = [s for s in samples if s.category == cat]
        print(f"\n  [{cat}] ({len(cat_samples)} samples)")
        for vname, vcfg in VARIANTS:
            coll = baseline_coll if vcfg["coll"] == "baseline" else optimized_coll
            cat_p5 = []
            for cs in cat_samples:
                r = _run_variant(cs, coll,
                    use_hybrid=vcfg["use_hybrid"],
                    use_query_plan=vcfg["use_query_plan"],
                    use_rerank=vcfg["use_rerank"],
                    use_source_priority=vcfg["use_source_priority"])
                cat_p5.append(r["precision_at_k"])
            avg_p5 = round(sum(cat_p5) / len(cat_p5), 4) if cat_p5 else 0
            print(f"    {vname:<25} P@5 = {avg_p5:.4f}")

    # ── Component contribution ──
    print("\n" + "=" * 80)
    print("COMPONENT CONTRIBUTION")
    print("=" * 80)
    components = [
        ("Hybrid retrieval (vector + keyword + RRF)", variant_summaries[1].get("precision_delta", 0)),
        ("Multi-query planning (plan_queries)", variant_summaries[2].get("precision_delta", 0)),
        ("Parent/child chunk index", variant_summaries[3].get("precision_delta", 0)),
        ("Reranking", variant_summaries[4].get("precision_delta", 0)),
        ("Source priority sorting", variant_summaries[5].get("precision_delta", 0)),
    ]
    total = variant_summaries[-1]["precision_at_k"] - variant_summaries[0]["precision_at_k"]
    for name, delta in components:
        pct = (delta / total * 100) if total > 0 else 0
        print(f"  {name:<45} ΔP@5 = {delta:+.4f}  ({pct:+.1f}% of total)")
    print(f"\n  Total uplift: {total:+.4f}  ({variant_summaries[0]['precision_at_k']:.4f} → {variant_summaries[-1]['precision_at_k']:.4f})")

    out = {"variants": variant_summaries, "components": components, "total_uplift": total,
           "per_sample": {v: rows for v, rows in all_rows.items()}}
    out_path = REPO_ROOT / "scripts" / "ablation_retrieval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_ablation()
