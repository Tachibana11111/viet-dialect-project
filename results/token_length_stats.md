# Token Length Statistics — ViDia2Std Central

| Split | Field | Min | Max | Avg | P50 | P95 | P99 |
|-------|-------|-----|-----|-----|-----|-----|-----|
| train | dialect | 5 | 60 | 15.4 | 14 | 25 | 31 |
| train | standard | 5 | 58 | 14.8 | 14 | 25 | 30 |
| dev | dialect | 7 | 63 | 15.2 | 14 | 26 | 32 |
| dev | standard | 6 | 60 | 14.4 | 13 | 25 | 30 |
| test | dialect | 6 | 39 | 15.7 | 15 | 26 | 31 |
| test | standard | 5 | 38 | 15.2 | 14 | 26 | 30 |

## Kết luận
- max_length = **64** cho PhoBERT classifier (max=60, p99≤60)
- max_length = **64** cho ViT5 seq2seq (an toàn cho cả input lẫn target)
- Không cần truncation — toàn bộ dataset nằm trong giới hạn
