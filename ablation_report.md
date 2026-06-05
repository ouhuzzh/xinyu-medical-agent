# Retrieval Ablation Study Report

| Variant | Precision@5 | MRR | Avg Latency (ms) | Vector (ms) | Keyword (ms) | Rerank (ms) | Samples |
|---------|-------------|-----|-------------------|-------------|--------------|-------------|---------|
| baseline | 0.3810 | 0.0000 | 1062.5 | 353.3 | 0.0 | 161.5 | 21 |
| no_rewrite | 0.3810 | 0.0000 | 1030.5 | 365.5 | 0.0 | 127.8 | 21 |
| no_hybrid | 0.3810 | 0.0000 | 848.7 | 359.1 | 0.0 | 133.3 | 21 |
| no_rerank | 0.3810 | 0.0000 | 920.3 | 366.1 | 0.0 | 0.0 | 21 |
| no_rewrite+no_rerank | 0.3810 | 0.0000 | 940.2 | 382.5 | 0.0 | 0.0 | 21 |

## Marginal Contribution vs Baseline

| Variant | Δ Precision@5 | Δ MRR | Δ Latency (ms) |
|---------|---------------|-------|----------------|
| no_rewrite | +0.0000 | +0.0000 | -32.0 |
| no_hybrid | +0.0000 | +0.0000 | -213.8 |
| no_rerank | +0.0000 | +0.0000 | -142.2 |
| no_rewrite+no_rerank | +0.0000 | +0.0000 | -122.3 |