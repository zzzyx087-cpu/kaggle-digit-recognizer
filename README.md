# Kaggle Digit Recognizer

PyTorch pipeline for the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition: classify 28×28 grayscale handwritten digits (0–9), starting from logistic regression and progressing through MLP and CNN experiments.

## Key Results

| Metric | Value |
| --- | --- |
| Best validation accuracy | **99.12%** |
| Kaggle public score | **0.99246** (99.246% public accuracy) |
| Best model | `CNN_BN` + Adam + CosineAnnealingLR + RandomAffine |
| Training epochs (final run) | 20 |
| Framework | PyTorch |

Validation accuracy is measured on a held-out 10% split of `train.csv` (seed 42). Kaggle public accuracy is the competition leaderboard score on the official test set.

## Model Development

Experiments were run in this order:

1. **Logistic Regression** — single linear layer `Linear(784, 10)`
2. **MLP** — `784 → 128 → ReLU → 10`
3. **MLP (wider)** — `784 → 256 → ReLU → 10`
4. **Deep MLP** — `784 → 256 → ReLU → 128 → ReLU → 10`
5. **CNN + SGD** — two conv blocks + linear classifier
6. **CNN + Adam** — same CNN, Adam optimizer
7. **CNN + CosineAnnealingLR** — Adam with learning-rate schedule
8. **CNN + BatchNorm** — `CNN_BN` architecture
9. **CNN + Data Augmentation** — RandomAffine on the base CNN
10. **Combined optimization** — `CNN_BN` + Adam + CosineAnnealingLR + RandomAffine

## Experiment Results

Validation accuracy from `experiments.md` (single-run results; small gaps should not be treated as statistically conclusive):

| Experiment | Model / Setup | Val Acc. | Kaggle Public |
| --- | --- | ---: | ---: |
| 1 | Logistic Regression (SGD, 10 epochs) | 89.90% | — |
| 2 | MLP 128 (SGD, 100 epochs) | 96.60% | — |
| 3 | MLP 256 (SGD, 100 epochs) | 96.79% | — |
| 4 | Deep MLP (SGD, 100 epochs) | 97.43% | — |
| 5 | CNN + SGD (10 epochs) | 96.98% | — |
| 6 | CNN + Adam (10 epochs) | 98.52% | — |
| 6b | CNN + CosineAnnealingLR (20 epochs) | 98.64% | — |
| 7 | CNN + BatchNorm (20 epochs) | 98.69% | — |
| 8 | CNN + RandomAffine | 99.10% | — |
| 9 | CNN_BN + Adam + Scheduler + Augmentation (20 epochs) | **99.12%** | **99.246%** |

## Final CNN Architecture

Submission model: `CNN_BN` in `src/model.py`.

```text
Conv2d(1 → 32, kernel_size=3)
BatchNorm2d(32)
ReLU
MaxPool2d(2)
Conv2d(32 → 64, kernel_size=3)
BatchNorm2d(64)
ReLU
MaxPool2d(2)
Flatten
Linear(1600 → 10)
```

Input shape: `(N, 1, 28, 28)`. Output: class logits of shape `(N, 10)`.

Final training setup (`src/train9_CNN_Combined.py`):

- Optimizer: Adam (`lr=0.001`)
- Scheduler: CosineAnnealingLR (`T_max=20`, `eta_min=1e-5`)
- Augmentation: `RandomAffine(degrees=10, translate=(0.08, 0.08))`
- Loss: CrossEntropyLoss
- Batch size: 64
- Epochs: 20
- Checkpoint: `outputs/best_cnn_combined.pth` (best validation epoch)

## Project Structure

```text
kaggle-digit-recognizer/
├── README.md
├── experiments.md
├── design_doc.md
├── requirements.txt
├── .gitignore
├── data/                      # not tracked (download from Kaggle)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/                   # not tracked (model checkpoints)
│   └── best_cnn_combined.pth
├── submission.csv             # not tracked
└── src/
    ├── dataset.py             # load / normalize / train-val split
    ├── model.py               # LogisticRegression, MLP*, CNN, CNN_BN
    ├── train1_Linear.py
    ├── train2_MLP.py
    ├── train3_MLP_256.py
    ├── train4_MLP_2linears.py
    ├── train5_CNN.py
    ├── train6_CNN_Scheduler.py
    ├── train7_CNN_BatchNorm.py
    ├── train8_CNN_Augmentation.py
    ├── train9_CNN_Combined.py
    └── submit.py              # write submission.csv
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Data comes from the Kaggle Digit Recognizer competition and is **not** included in this repository.

1. Download `train.csv` and `test.csv` from the competition page.
2. Place them under `data/`:

```text
data/train.csv
data/test.csv
```

`dataset.py` uses a 90/10 train/validation split with `torch.Generator().manual_seed(42)`.

## Training

Run all commands from the **project root** (paths like `data/` and `outputs/` are relative to the root).

```bash
# Baseline models
python src/train1_Linear.py
python src/train2_MLP.py
python src/train3_MLP_256.py
python src/train4_MLP_2linears.py

# CNN experiments
python src/train5_CNN.py
python src/train6_CNN_Scheduler.py
python src/train7_CNN_BatchNorm.py
python src/train8_CNN_Augmentation.py

# Final combined training (saves outputs/best_cnn_combined.pth)
python src/train9_CNN_Combined.py
```

## Inference and Kaggle Submission

Requires `data/test.csv` and `outputs/best_cnn_combined.pth`.

```bash
python src/submit.py
```

This writes `submission.csv` with columns:

| Column | Description |
| --- | --- |
| `ImageId` | 1-based row index |
| `Label` | Predicted digit (0–9) |

Upload `submission.csv` on the Kaggle competition page.

## Key Findings

- **CNN vs MLP.** Convolutional layers use spatial structure in the digit images. With comparable or fewer epochs, CNNs reached higher validation accuracy than flat MLPs once training was tuned (Adam and later improvements).
- **Adam.** Switching from SGD to Adam on the same CNN raised validation accuracy from 96.98% to 98.52% in this project’s recorded runs.
- **Batch Normalization.** Adding BatchNorm after each convolution slightly improved the Adam baseline (98.52% → 98.69%). The gain was small and was only measured once.
- **Data augmentation.** RandomAffine produced the largest single jump among the later CNN runs (to 99.10% validation), with higher training loss but better validation accuracy—consistent with stronger regularization.
- **Combined optimization.** BatchNorm + Adam + CosineAnnealingLR + RandomAffine reached 99.12% validation and 99.246% Kaggle public accuracy. The 0.02% validation edge over augmentation alone is small and should not be over-interpreted.

## Future Improvements

- Stronger augmentation (e.g. elastic distortion, random erasing)
- Broader hyperparameter search and multi-seed evaluation
- Ensembles of CNN checkpoints or architectures

## Notes on Reproducibility

- Importing `src/model.py` currently executes top-level demo prints and loads `train_loader` as a side effect. Training scripts still run from the project root as shown above.
- `train2_MLP.py` and `train3_MLP_256.py` import `tqdm` but do not use it.
- See `experiments.md` for per-run losses and written observations.
