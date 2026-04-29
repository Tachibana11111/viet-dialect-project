# Phase 4 Results — End-to-End Pipeline

## Pipeline Architecture
Input: Cau phuong ngu Trung Bo
↓ Edge case check
↓ PhoBERT + Lexicon Ensemble (~18ms)
↓ ViT5-base fine-tuned (~553ms)
Output: Tieng Viet chuan

## End-to-End Metrics (100 samples)
| Metric | Value |
|--------|-------|
| BLEU | 0.7792 |
| ROUGE-L | 0.9269 |
| Latency median | ~579ms |
| Latency P95 | ~1398ms |

## Component Performance
| Component | Metric | Value |
|-----------|--------|-------|
| PhoBERT classifier | F1 macro | 0.5207 |
| ViT5 normalizer | BLEU | 0.7937 |
| Full pipeline | BLEU | 0.7792 |

## Edge Cases Handled
1. too_short: < 2 tokens → return input as-is
2. already_standard: no dialect markers → bypass ViT5

## Delta Phase3 → Phase4
- BLEU   : -0.0145 (expected — edge cases + 100 sample variance)
- ROUGE-L: -0.0038

## Artifacts
- figures/phase4_comparison.png
- figures/phase4_summary.png
