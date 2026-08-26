# AG-DBSwin: Attribution-Guided Dual-Branch Swin Transformer

AG-DBSwin is a research pipeline for five-class diabetic retinopathy (DR) grading from retinal fundus photographs. A base Swin Transformer learns visual features from RGB images, Integrated Gradients produces lesion-attribution maps, and a second Swin branch learns from those maps alongside the RGB branch.

The repository contains the complete experiment notebook, quantitative results, per-sample predictions, trained checkpoints, generated heatmaps, and qualitative inference visualizations.

> **Research-use notice:** This project is an experimental research implementation. It is not a medical device and must not be used for clinical diagnosis or treatment decisions.

## Method overview

The executable pipeline in [`AG-DBSwin.ipynb`](AG-DBSwin.ipynb) follows these stages:

1. Resize fundus images to `224 × 224` and apply ImageNet normalization.
2. Train an ImageNet-pretrained `swin_tiny_patch4_window7_224` base classifier.
3. Generate Integrated Gradients heatmaps using the trained base model.
4. Train a dual-branch model with one Swin backbone for RGB images and one for heatmaps.
5. Concatenate the two 768-dimensional feature vectors and classify through a `1536 → 512 → 256 → 5` head.
6. Evaluate with flip-based test-time augmentation (TTA), accuracy, and Quadratic Weighted Kappa (QWK).
7. Run component ablations and stratified five-fold experiments.

Important implementation details include paired spatial augmentation for images and heatmaps, ordinal label smoothing, class-weighted focal loss, AdamW, gradient clipping, cosine learning-rate scheduling, validation-QWK checkpointing, and early stopping.

The earlier loopback fine-tuning stage is intentionally excluded. Its measured gain was small relative to its computational cost, so the Stage 4 dual-branch checkpoint is treated as the final model.

## Dataset

Experiments use the [APTOS-2019 dataset on Kaggle](https://www.kaggle.com/datasets/mariaherrerot/aptos2019), a split version of the APTOS 2019 Blindness Detection data.

The Kaggle data card describes 3,662 fundus photographs collected under varied imaging conditions and reviewed by trained clinicians. Labels follow five diabetic-retinopathy severity grades:

| Label | Grade |
|---:|---|
| 0 | No DR |
| 1 | Mild DR |
| 2 | Moderate DR |
| 3 | Severe DR |
| 4 | Proliferative DR |

The raw dataset is not included in this repository. At the time this README was prepared, the supplied Kaggle mirror listed its license as **Unknown**. Download and use the data only in accordance with the Kaggle dataset page and any applicable APTOS competition terms.

Download with the Kaggle CLI:

```bash
kaggle datasets download -d mariaherrerot/aptos2019
```

Arrange the extracted split files for the notebook as follows. If the downloaded directory names differ, rename them or update the driver paths in the notebook.

```text
aptos/
├── train.csv
├── valid.csv
├── test.csv
├── train_images/
├── val_images/
└── test_images/
```

Each CSV must contain `id_code` and `diagnosis` columns. The notebook appends `.png` to each `id_code` when constructing image paths.

## Results

Accuracy and QWK are reported separately for each experimental protocol. The main full-budget run, reduced-budget ablations, and reduced-budget cross-validation runs should not be interpreted as identical training settings.

### Main single-split run

The main notebook driver uses 15 base-model epochs and 12 dual-branch epochs.

| Protocol | Accuracy | QWK |
|---|---:|---:|
| Original train/validation/test split | 0.9044 | 0.9518 |

### Ablation study

The ablation sweep uses one seed (`seed=0`) and a shared reduced budget of 6 base epochs and 5 dual-branch epochs.

| Configuration | Accuracy | QWK |
|---|---:|---:|
| Full model | **0.9098** | 0.9418 |
| Without augmentation | 0.8443 | 0.9079 |
| Without ordinal smoothing | 0.8770 | 0.9269 |
| Without dual branch | 0.8142 | 0.9096 |
| Without TTA | 0.8907 | 0.9384 |
| Without class weighting | 0.9071 | **0.9472** |
| Without cosine scheduler | 0.9016 | 0.9297 |

The largest accuracy reduction occurs when the dual branch is removed, followed by removal of augmentation and ordinal smoothing. Because each ablation was run with a single seed, small differences should be treated as sensitivity evidence rather than robust variance estimates.

Source: [`results/ablation_results.csv`](results/ablation_results.csv)

### Five-fold summary

The full-model cross-validation experiment also uses the reduced 6/5-epoch budget and evaluates every fold on the held-out test set.

| Metric | Mean | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Accuracy | 0.8934 | 0.0122 | 0.8716 | 0.8989 |
| QWK | 0.9410 | 0.0082 | 0.9301 | 0.9492 |

The original full-budget single-split result lies above the reduced-budget five-fold mean for both metrics. See [`results/cv_results.csv`](results/cv_results.csv), [`results/cv_summary.csv`](results/cv_summary.csv), and [`results/original_vs_cv_comparison.csv`](results/original_vs_cv_comparison.csv).

## Qualitative inference outputs

The [`inference output`](inference%20output/) directory contains 12 previously generated four-panel attribution visualizations showing the source image and saliency outputs labelled ResNet18, ResNet50, and AG-DBSwin (`Ours`). These panels are qualitative illustrations; quantitative conclusions should be based on the result tables above.

![Qualitative attribution example](inference%20output/fig_0.png)

![Additional qualitative attribution example](inference%20output/fig_1.png)

## Repository structure

```text
.
├── AG-DBSwin_v2.ipynb       # Complete training and evaluation workflow
├── inference output/        # 12 qualitative attribution panels
└── results/
    ├── heatmaps/             # 11 per-run ZIP archives containing .npy maps
    ├── models/               # 12 per-experiment checkpoint ZIP archives
    ├── *_preds.json          # Per-sample labels and predictions
    ├── *.csv                 # Ablation, CV, summary, and comparison tables
    ├── all_results.xlsx      # Consolidated spreadsheet
    ├── manifest.csv          # Artifact paths, sizes, counts, and hashes
    └── SHA256SUMS.txt        # SHA-256 integrity checks
```

The archived research package contains:

| Category | Archives/files | Archived members | Size |
|---|---:|---:|---:|
| Heatmap ZIPs | 11 | 40,282 `.npy` files | 4.183 GiB |
| Model ZIPs | 12 | 23 `.pth` checkpoints | 3.268 GiB |
| Tables and predictions | 16 | 16 files | < 1 MiB |

The complete results package is approximately 7.45 GiB. Large ZIP archives are tracked using Git LFS.

## Setup

### 1. Clone the repository and download LFS objects

```bash
git lfs install
git clone https://github.com/RajiiLakshan/AG-DBSwin.git
cd AG-DBSwin
git lfs pull
```

### 2. Create an environment

Python 3.9 was used for the recorded notebook environment. Create and activate a virtual environment, then install the CUDA-compatible PyTorch and torchvision build recommended by the [official PyTorch installer](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the appropriate torch and torchvision build for your CUDA version.
pip install jupyter
jupyter lab
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate torch and torchvision build for your CUDA version.
pip install jupyter
jupyter lab
```

Open `AG-DBSwin_v2.ipynb`. Its environment cell installs the remaining Python packages (`timm`, `captum`, `thop`, scikit-learn, seaborn, pandas, NumPy, Pillow, matplotlib, openpyxl, and tqdm).

### 3. Run the experiments

Run the notebook sections in order:

1. environment, imports, datasets, and model definitions;
2. main driver for the full-budget single-split experiment;
3. ablation sweep;
4. cross-validation and summary;
5. comparison and Excel export.

Running the entire notebook retrains multiple Swin models and regenerates tens of thousands of heatmaps. A CUDA-capable GPU and substantial execution time and storage are required.

## Using the archived artifacts

Each heatmap directory and model run is packaged independently so that individual experiments can be downloaded and restored without extracting the entire artifact collection.

Examples:

```powershell
Expand-Archive results\heatmaps\full_model_seed0_hmv1.zip -DestinationPath restored\full_model_seed0_hmv1
Expand-Archive results\models\full_model_seed0_models.zip -DestinationPath restored\full_model_seed0_models
```

Verify all downloaded artifacts against the recorded hashes:

```powershell
$root = Resolve-Path results
Get-Content results\SHA256SUMS.txt | ForEach-Object {
    if ($_ -match '^([0-9a-f]{64})  (.+)$') {
        $path = Join-Path $root $Matches[2]
        $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $Matches[1]) { throw "Checksum mismatch: $path" }
    }
}
```

Refer to [`results/manifest.csv`](results/manifest.csv) for the experiment name, member count, byte size, and SHA-256 value of every published artifact.

## Reproducibility notes

- Seeds are set for Python, NumPy, and PyTorch in the ablation/CV runner.
- Best checkpoints are selected using validation QWK rather than accuracy.
- Prediction JSON files preserve `y_true` and `y_pred` for independent metric checks.
- CV results are saved after every completed run so interrupted sweeps can resume.
- Generated heatmaps and checkpoints are retained to support exact downstream reruns without regenerating every intermediate artifact.
- Hardware, CUDA kernels, data-loader workers, and library versions can still introduce nondeterminism; exact bitwise reproduction is not guaranteed.

## Authors

RajaRajeshwari G School of Computer Science Engineering and Information Systems Vellore Institute of Technology Vellore, Tamil Nadu, India

Chemmalar Selvi G School of Computer Science Engineering and Information Systems Vellore Institute of Technology Vellore, Tamil Nadu, India

Corresponding Author: Chemmalar Selvi G


## Acknowledgements

This work uses the APTOS 2019 Blindness Detection data distributed through Kaggle. The dataset is associated with the Asia Pacific Tele-Ophthalmology Society and fundus-image collection efforts described by the Kaggle data card, including Aravind Eye Hospital in India.
