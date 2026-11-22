# SurfaceGuard AI: Industrial Surface Anomaly Segmentation

BSDA2001P Introduction to DL and GenAI Project (Term 3, 2026) · Dasam Sai Kumar · Roll no. 23f3001567

Pixel-level segmentation of four industrial surface-defect types (defect_a to defect_d) in 128 × 800 grayscale images.
Kaggle competition: *industrial-surface-defect-detection*. Score = 0.70 · foreground Dice + 0.20 · false-positive score + 0.10 · empty-prediction accuracy.

## Results

| Model | Type | Params (M) | Val score | Foreground Dice | Macro F1 |
|---|---|---|---|---|---|
| U-Net from scratch | scratch | 7.8 | 0.6808 | 0.5709 | 0.5090 |
| U-Net + ResNet34 | pretrained | 24.4 | 0.7003 | 0.5900 | 0.5424 |
| U-Net + EfficientNet-B0 | pretrained | 5.8 | 0.7306 | 0.6271 | 0.5950 |
| U-Net + EfficientNet-B2 | pretrained | 9.5 | 0.7330 | 0.6321 | 0.6041 |
| U-Net + EfficientNet-B3 | pretrained | 12.6 | 0.7347 | 0.6312 | 0.6089 |
| U-Net + EfficientNet-B2 #2 (shift/zoom aug.) | pretrained | 9.5 | 0.7357 | 0.6367 | 0.6088 |
| **Ensemble B0 + B3 + B2 #2** (tuned rules + 4-view TTA) | – | – | **0.7554** | 0.6617 | 0.6104 |

**Kaggle leaderboard: 0.69878** (rank 264; cutoff 0.65). Validation score is 0.057 higher than the leaderboard score because the rules were tuned on validation and the test set contains more and harder defects.

## Pipeline

1. **Data:** `train.csv` (image_id, defect_type, mask_rle) → decode RLE to 4 masks per image; 80/20 split stratified by defect pattern (seed 42).
2. **Preprocessing / augmentation:** grayscale, normalisation, flips, gamma, contrast, brightness, noise (+ shift/zoom for one model); weighted sampler for rare defects.
3. **Models:** U-Net from scratch, ResNet34-U-Net, EfficientNet-B0/B2/B3-U-Nets (ImageNet weights, first conv adapted to 1 channel).
4. **Training:** loss = BCE + ½ batch-Dice + ½ per-pair Dice; AdamW, warm-up + cosine, EMA, mixed precision; every model logged to Weights & Biases.
5. **Post-processing:** per-defect threshold, minimum area and confidence gate tuned on the official metric; 4-view flip TTA; probability averaging of three EfficientNets.
6. **Submission:** masks encoded to RLE in the order of `sample_submission.csv`.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/` | milestone notebooks and the final Kaggle notebook (all six models, ensemble, error analysis) |
| `scripts/rle_metric.py` | RLE encode/decode, Dice, post-processing and the official score, with self-tests |
| `report/` | final project report (PDF) |
| `requirements.txt` | pinned package versions used on Kaggle |

## How to run

1. On Kaggle: add the competition data, set **GPU T4**, **Internet ON**, and add the secret `WANDB_API_KEY`.
2. Open the final notebook and *Save & Run All*. Training of all six models takes about 5 hours on a T4 (`QUICK_TEST = True` runs a 10-minute dry run).
3. Check the metric helpers locally: `pip install -r requirements.txt && python scripts/rle_metric.py`.

Experiment tracking: Weights & Biases project `23f3001567-t32026` (one run per model plus comparison runs).
