# Vietnamese Central Dialect Normalization

Fine-tuning PhoBERT and ViT5 for bidirectional normalization between Vietnamese
Central dialects and standard Vietnamese.

**Live demo:** https://huggingface.co/spaces/tachibana11111/viet-dialect-demo  
**Models:** https://huggingface.co/tachibana11111/viet-dialect-normalizer

---

## Overview

Vietnamese Central dialects (12 provinces from Thanh Hoa to Khanh Hoa) contain
vocabulary and grammatical structures that cause standard NLP systems to fail.
This project builds an end-to-end pipeline that:

1. Identifies the dialect group (Thanh-Nghe-Tinh / Binh-Tri-Thien / Nam-Trung-Bo)
2. Normalizes dialect text to standard Vietnamese — or generates dialect from standard
3. Handles edge cases (already-standard input, very short input)

---

## Results

**ViT5 Normalizer** — evaluated on ViDia2Std Central test set (1,066 samples, beam=4):

| Direction | BLEU | ROUGE-L |
|---|---|---|
| dialect to standard | 0.7937 | 0.9307 |
| standard to dialect | 0.6717 | 0.8987 |

Comparison with ViDia2Std paper baselines (all regions):

| Model | BLEU | ROUGE-L |
|---|---|---|
| mBART-large-50 (paper) | 0.8166 | 0.9384 |
| ViT5-base (this project, Central only) | 0.7937 | 0.9307 |

**PhoBERT Classifier** — evaluated on ViMD Central test set (296 samples):

| Metric | Value |
|---|---|
| Accuracy | 52.03% |
| F1 Macro | 0.5207 |

Note: Classifier performance is bounded by ViMD's formal radio speech domain.
The normalizer (ViT5) operates independently and is not affected by classifier errors.

---

## Dataset

| Dataset | Paper | Samples Used | Task |
|---|---|---|---|
| ViDia2Std | AAAI 2026 | 9,029 Central pairs | ViT5 training |
| ViMD | EMNLP 2024 | 2,959 Central text samples | PhoBERT training |

Central scope: 12 provinces grouped into 3 dialect clusters.

| Group | Provinces |
|---|---|
| Thanh-Nghe-Tinh | Thanh Hoa, Nghe An, Ha Tinh |
| Binh-Tri-Thien | Quang Binh, Quang Tri, Thua Thien Hue |
| Nam-Trung-Bo | Da Nang, Quang Nam, Quang Ngai, Binh Dinh, Phu Yen, Khanh Hoa |

---

## Architecture

```
Input: dialect text
    |
    v
[Edge Case Check]
    |-- too short (<2 tokens)     --> return as-is
    |-- no dialect markers found  --> return as-is (bypass ViT5)
    |
    v
[PhoBERT + Lexicon Ensemble]  ~18ms
    |-- classify into 3 dialect groups
    |-- lexicon: frequency-ratio markers from ViDia2Std
    |-- NTB-aware guard: trust PhoBERT when score > 0.5 for NTB
    |
    v (metadata only, does not affect normalization)
[ViT5 Fine-tuned]  ~553ms
    |-- prefix: "dialect2std: {input}"
    |-- or:     "std2dialect: {input}"
    |-- beam=4 (evaluation) / beam=1 (production)
    |
    v
Output: standard Vietnamese
```

End-to-end latency: ~579ms median on Tesla T4 GPU.

---

## Repository Structure

```
viet-dialect-project/
|
|-- data/
|   |-- raw/
|   |   |-- vimd_central_text.csv          # ViMD Central text cache (stream once)
|   |   |-- vimd_central_distribution.csv  # Province distribution
|   |   `-- vimd_all_distribution.csv
|   |-- processed/
|   |   `-- combined_classifier_data.csv   # 2,959 samples, 3 groups
|   `-- splits/
|       |-- vidia_train/dev/test.csv       # ViDia2Std splits
|       `-- clf_train/val/test.csv         # Classifier splits
|
|-- models/
|   |-- phobert-classifier/best/           # PhoBERT checkpoint
|   `-- vit5-normalizer/best/              # ViT5 checkpoint
|
|-- figures/
|   |-- phase1_summary.png                 # Data distribution
|   |-- phobert_confusion_matrix.png       # Classifier confusion matrix
|   |-- phase4_comparison.png              # BLEU comparison chart
|   `-- phase4_summary.png                 # Pipeline summary
|
|-- results/
|   |-- distribution_report.md
|   |-- token_length_stats.md
|   |-- phase2_results.md                  # PhoBERT results
|   |-- phase3_results.md                  # ViT5 results
|   |-- phase4_results.md                  # Pipeline results
|   |-- lexicon_config.json                # Dialect lexicon
|   `-- vit5_test_predictions.csv          # Full test predictions
|
`-- notebooks/
    `-- load_data.py                       # Data loading utility
```

---

## Quickstart

### Requirements

```bash
pip install transformers sentencepiece huggingface_hub torch numpy
```

### Normalize dialect to standard

```python
import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM

MODEL_REPO = "tachibana11111/viet-dialect-normalizer"

spiece_path = hf_hub_download(repo_id="VietAI/vit5-base", filename="spiece.model")
sp = spm.SentencePieceProcessor()
sp.Load(spiece_path)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO, subfolder="vit5-normalizer")
model.eval()

def normalize(text, num_beams=4):
    ids = sp.EncodeAsIds("dialect2std: " + text) + [1]
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(input_ids, max_length=80,
                             num_beams=num_beams, early_stopping=True)
    result = [i for i in out[0].tolist() if i > 1]
    return sp.DecodeIds(result)

print(normalize("Choa chua chod mua mo ca."))
# "Toi chua thay mua dau ca."

print(normalize("Con tru ni beo qua."))
# "Con trau nay beo qua."
```

### Classify dialect group

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ID2LABEL = {0: "Thanh-Nghe-Tinh", 1: "Binh-Tri-Thien", 2: "Nam-Trung-Bo"}

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
classifier = AutoModelForSequenceClassification.from_pretrained(
    "tachibana11111/viet-dialect-normalizer",
    subfolder="phobert-classifier",
    num_labels=3
)
classifier.eval()

def classify(text):
    enc = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=64, padding="max_length")
    with torch.no_grad():
        probs = F.softmax(classifier(**enc).logits, dim=-1).squeeze()
    return ID2LABEL[int(torch.argmax(probs))]

print(classify("Choa chua chod mua."))
# "Thanh-Nghe-Tinh"
```

### Use via Gradio Client (no download required)

```python
from gradio_client import Client

client = Client("tachibana11111/viet-dialect-demo")
result = client.predict("Con tru ni beo qua.", api_name="/normalize_dialect")
```

---

## Training

Training was conducted on Google Colab with Tesla T4 GPU.
All notebooks and scripts are in the `notebooks/` directory.

**PhoBERT Classifier key settings:**

| Parameter | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 5 (best: epoch 4) |
| Class weights | [1.3156, 1.3174, 0.6753] |
| Loss | WeightedCrossEntropyLoss |

**ViT5 Normalizer key settings:**

| Parameter | Value |
|---|---|
| Learning rate | 3e-4 |
| Batch size | 16 |
| Epochs | 10 (best: epoch 3) |
| Task prefix | dialect2std / std2dialect |
| Max length | 80 tokens |

---

## Limitations

- The PhoBERT classifier F1=0.5207 is bounded by the ViMD dataset domain
  (formal radio broadcast speech). It does not affect normalization quality
  since ViT5 operates as a single model for all Central dialects.
- ViT5 may produce incorrect output for highly idiomatic expressions or
  rare vocabulary not present in ViDia2Std training data.
- Only Central Vietnamese dialects are supported.

---

## Citation

```bibtex
@inproceedings{vidia2std2026,
  title     = {ViDia2Std: A Parallel Corpus and Methods for
                Low-Resource Vietnamese Dialect-to-Standard Translation},
  booktitle = {Proceedings of AAAI 2026}
}

@inproceedings{vimd2024,
  title     = {ViMD: Vietnamese Multi-Dialect Dataset},
  booktitle = {Proceedings of EMNLP 2024}
}
```

---

## License

Models and code: MIT  
ViDia2Std dataset: CC-BY-NC-4.0  
ViMD dataset: CC-BY-NC-ND-4.0
