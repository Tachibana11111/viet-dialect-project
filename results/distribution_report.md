# Dataset Distribution Report

## ViDia2Std
| Split | Total | Central | Northern | Southern |
|-------|-------|---------|----------|----------|
| Train | 10,870 | 7,197 | 758 | 2,915 |
| Dev   | 1,184  | 769   | -   | -     |
| Test  | 1,603  | 1,067 | -   | -     |
| **Total** | **13,657** | **9,033** | - | - |

> Source: Facebook comments, annotated by native speakers
> License: CC-BY-NC-4.0
> Paper: AAAI-26

---

## ViMD (Central only, text metadata)
| Nhóm | Tỉnh | Samples |
|------|------|---------|
| Thanh-Nghe-Tinh | ThanhHoa, NgheAn, HaTinh | 754 |
| Binh-Tri-Thien  | QuangBinh, QuangTri, ThuaThienHue | 753 |
| Nam-Trung-Bo    | DaNang, QuangNam, QuangNgai, BinhDinh, PhuYen, KhanhHoa | 1,469 |
| **Total** | 12 tỉnh | **2,976** |

> Source: Radio broadcast transcripts, 63 tỉnh
> License: CC-BY-NC-ND-4.0
> Paper: EMNLP-24

---

## Ghi chú imbalance
- ViMD Central: Nam-Trung-Bo chiếm 49.4% do có 6 tỉnh vs 3 tỉnh/nhóm kia
- Cần oversample Thanh-Nghe-Tinh và Binh-Tri-Thien ở Phase 2
- ViDia2Std Central: 9,033 samples — đủ để fine-tune ViT5
