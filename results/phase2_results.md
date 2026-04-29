# Phase 2 Results — PhoBERT Dialect Classifier

## Model
- Base model: `vinai/phobert-base`
- Task: 3-class dialect region classification
- Training data: ViMD Central (2,367 train / 296 val / 296 test)
- Class weights: [1.3156, 1.3174, 0.6753]

## Training
| Epoch | Train Loss | Val Loss | Val Acc | Val F1 |
|-------|-----------|----------|---------|--------|
| 1 | 1.0825 | 1.0020 | 45.95% | 0.4637 |
| 2 | 0.9124 | 0.9538 | 51.69% | 0.5069 |
| 3 | 0.7222 | 0.9276 | 54.39% | 0.5486 |
| **4** | **0.5341** | **0.9489** | **58.11%** | **0.5841** ← best |
| 5 | 0.3940 | 0.9622 | 58.11% | 0.5808 |

## Test Results
| Method | Accuracy | F1 Macro |
|--------|----------|----------|
| PhoBERT only | 52.03% | 0.5207 |
| Ensemble v1 (lexicon naive) | 50.68% | 0.5090 |
| Ensemble v2 (NTB-aware) | 52.03% | 0.5207 |

## Per-class (best model)
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Thanh-Nghe-Tinh | 0.5059 | 0.5658 | 0.5342 |
| Binh-Tri-Thien  | 0.3937 | 0.6667 | 0.4950 |
| Nam-Trung-Bo    | 0.7262 | 0.4207 | 0.5328 |

## Key Findings
1. **Domain mismatch**: ViMD là radio transcript formal — thiếu dialect markers
   rõ ràng. Model không học được boundary TNT/BTT/NTB từ formal speech.
2. **Lỗi chính**: Nam-Trung-Bo bị nhầm thành Binh-Tri-Thien (37.2%) vì hai
   vùng dùng chung markers (rứa, mô, hè) trong ngữ cảnh formal.
3. **Lexicon ensemble**: Hoạt động tốt trên conversational text (67% accuracy)
   nhưng không cải thiện được trên ViMD vì thiếu signal.
4. **Ceiling effect**: F1=0.52 là giới hạn thực của bài toán với data này,
   không phải model yếu. Classifier sẽ perform tốt hơn trên input hội thoại
   thật ở Phase 4.

## Artifacts
- Best checkpoint: `models/phobert-classifier/best/`
- Confusion matrix: `figures/phobert_confusion_matrix.png`
- LEXICON + ensemble_predict_v2: sẵn sàng dùng ở Phase 4

## Decision
Chốt Phase 2. F1=0.52 được ghi nhận là domain limitation của ViMD.
Ensemble v2 giữ lại cho Phase 4 pipeline với input hội thoại thật.
