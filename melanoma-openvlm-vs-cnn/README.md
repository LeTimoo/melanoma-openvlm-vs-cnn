# Open multimodal LLMs vs CNNs for melanoma detection in dermoscopy

Code and results for a course project at UiT The Arctic University of Norway
(Timothée Bruniaux, Karim Nasri, Anna Reedtz).

We compare two open-weight multimodal LLMs (Qwen2.5-VL-7B-Instruct and
LLaVA-OneVision-Qwen2-7B, zero-shot and few-shot) with two fine-tuned CNNs
(ResNet50, EfficientNet-B0) on 200 histopathology-confirmed HAM10000 images
(100 melanomas, 100 nevi).

## Main result

| Arm | Sensitivity | Specificity | AUC (95% CI) |
|---|---|---|---|
| ResNet50 | 0.67 | 0.82 | 0.78 (0.72–0.85) |
| EfficientNet-B0 | 0.80 | 0.62 | 0.81 (0.76–0.87) |
| Qwen2.5-VL zero-shot | 0.96 | 0.13 | 0.64 (0.56–0.72) |
| Qwen2.5-VL few-shot | 0.75 | 0.58 | 0.70 (0.62–0.77) |
| LLaVA-OV zero-shot | 0.08 | 0.97 | 0.66 (0.59–0.74) |
| LLaVA-OV few-shot | 0.02 | 1.00 | 0.67 (0.60–0.74) |

## Contents

- `melanoma_openvlm_vs_cnn.ipynb` — the full pipeline: data download, lesion-level split,
  CNN training, LLM scoring, statistics and figures. Runs on Google Colab with a T4 GPU.
- `results/split.csv` — the frozen split (image and lesion IDs, labels, set). No images.
  SHA-256 of the sorted, comma-joined test image IDs:
  `181b1764dd12d2e94159d86752e8b1814ddd7c848df8c71042a7356b0ec03a4a`
- `results/` — all predictions, LLM scores and sampled answers, metrics and statistical tests.
- `figures/` — figures used in the report.

## Reproducing

1. Open the notebook in Google Colab and select a T4 GPU runtime.
2. Run all cells. The notebook downloads HAM10000, rebuilds the split (seed 2026) and
   prints the test-list hash (compare it with the one above), then trains the CNNs and scores the LLMs.

Environment of the reported run: Python 3.13, PyTorch 2.14, timm 1.0, transformers 5.24,
scikit-learn 1.6, statsmodels 0.15 (see `results/environment.json`).

## Data

HAM10000 (Tschandl et al., 2018, doi:10.7910/DVN/DBW86T), licensed CC BY-NC 4.0.
Images are not redistributed here; only image identifiers are shared.
