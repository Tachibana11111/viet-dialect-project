# Phase 3 Results — ViT5 Dialect Normalizer

## Model
- Base model: `VietAI/vit5-base`
- Task: Seq2Seq dialect normalization (2 chieu)
- Training data: ViDia2Std Central (14,392 train / 1,534 dev / 2,132 test)
- Best epoch: 3/10 (early stopping by eval_loss)

## Training
| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 0.9747 | 0.7671 |
| 2 | 0.7626 | 0.7012 |
| **3** | **0.6140** | **0.6714** <- best |
| 4 | 0.4875 | 0.6783 |
| 5 | 0.3649 | 0.7712 |
| 6-10 | overfitting | 1.270 |

## Test Results
| Direction | BLEU | ROUGE-L |
|-----------|------|---------|
| dialect -> standard | 0.7937 | 0.9307 |
| standard -> dialect | 0.6717 | 0.8987 |

## So sanh voi paper ViDia2Std
| Model | BLEU | ROUGE-L |
|-------|------|---------|
| mBART-large (paper) | 0.8166 | 0.9384 |
| ViT5-base (ours, Central only) | 0.7937 | 0.9307 |

> Ket qua ngang paper du train tren Central only (khong phai 3 regions)

## Sample-level BLEU distribution
| Range | Count | % |
|-------|-------|---|
| 0-20  | 17  | 1.6% |
| 20-40 | 63  | 5.9% |
| 40-60 | 149 | 14.0% |
| 60-80 | 302 | 28.3% |
| 80-100 | 216 | 20.3% |
| Perfect (>=99) | 319 | 29.9% |

## Key Findings
1. BLEU=0.7937 dialect->std — ngang paper ViDia2Std voi ViT5-base
2. 29.9% perfect translations — model hoc duoc mapping rat tot
3. Loi chinh: tu dia phuong cuc hep va nghia bong
4. Chieu nguoc std->dialect BLEU=0.6717 — thap hon hop ly
5. Overfit tu epoch 4 — best model tai epoch 3

## Artifacts
- Best checkpoint : models/vit5-normalizer/best/
- Tokenizer       : SimpleT5Tokenizer (SentencePiece)
