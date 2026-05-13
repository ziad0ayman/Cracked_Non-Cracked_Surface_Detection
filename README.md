# 🏗️ Cracked vs. Non-Cracked Surface Detection

An end-to-end deep learning project that automatically detects structural cracks on concrete surfaces using an ensemble of five neural architectures. Includes a Streamlit dashboard for interactive inference and offline performance analysis.

## ✨ Features

- **5 diverse models** — FFNN, CNN, Column-wise BiLSTM, Multi-Directional RNN, DenseNet121 transfer learning
- **Ensemble voting** — all models predict and the majority decides the final verdict
- **Interactive dashboard** — upload an image and see per-model predictions, confidence bars, vote distribution, and a final ensemble verdict
- **Performance browser** — offline tab with accuracy, F1 per class, confusion matrices, and parameter counts for every model
- **Reproducible training** — each model has a standalone training script with `--resume` support, checkpointing, and class-weight balancing

## 📊 Dataset

[Cracked/Non-Cracked Surface Dataset](https://www.kaggle.com/datasets/geadalfa/cracked-non-cracked-surface-datasets) from Kaggle.

- **96,092 images** across 2 classes (Cracked / Non-Cracked)
- Split **80/15/5** into train/val/test
- Images resized to **224×224×3** for all models
- Class imbalance handled via `sklearn` balanced class weights (file-count based, deterministic)

## 🧠 Models

| Model | Architecture | Params | Batch Size |
|-------|-------------|--------|------------|
| **FFNN** | Flatten → Dense(512) → Dropout(0.157) → Dense(2) | 77.1M | 32 |
| *(Optuna search)* | *n_layers ∈ [1,2], n_units ∈ [512,1024], dropout ∈ [0.1,0.5], lr ∈ [1e-5,1e-2]* | | |
| **CNN** | 6× ConvBlock(Conv+BN+ReLU+MP) → GAP → Dense(128) → Dropout(0.254) | 6.3M | 64 |
| *(Optuna search)* | *n_blocks ∈ [3,6], base_filters ∈ [32,64], dense ∈ [128,384], dropout ∈ [0.1,0.5], lr ∈ [1e-5,1e-2]* | | |
| **Column RNN** | AvgPool → Permute → BiLSTM(128) → Dense(384) → Dropout(0.485) | 405K | 64 |
| *(Optuna search)* | *lstm_units ∈ [64,256], dense ∈ [128,512], dropout ∈ [0.1,0.5], lr ∈ [1e-5,1e-2]* | | |
| **MD-RNN** | Dual BiLSTM(rows + cols) → Concat → Dense(512) → Dropout(0.262) | 874K | 64 |
| *(Optuna search)* | *lstm_units ∈ [64,256], dense ∈ [128,512], dropout ∈ [0.1,0.5], lr ∈ [1e-5,1e-2]* | | |
| **DenseNet121** | DenseNet121 backbone → GAP → Dense(384) → BN → Dropout(0.423) | 7.4M | 64 |
| *(Optuna search)* | *dense ∈ [128,512], dropout ∈ [0.1,0.5], lr ∈ [1e-5,1e-2], unfreeze ∈ [10,40]* | | |

All models use:
- `categorical_crossentropy` loss with 2-class softmax output
- `Adam` optimizer with learning rates tuned via **Optuna** hyperparameter search (30–50 trials per model with pruning)
- Data augmentation (flip, rotation, zoom, contrast) embedded as a `Sequential` layer inside the model graph
- `image_dataset_from_directory` for the data pipeline (no legacy `ImageDataGenerator`)

> Every architecture — including layer counts, units, dropout rates, and learning rates — was discovered through Optuna Bayesian optimisation. The notebooks in `notebooks/model_architectures/` contain the full search histories.

## 📈 Results

| Model | Accuracy | F1 (Cracked) | F1 (Non-Cracked) | F1 (Macro) | Params |
|-------|----------|-------------|-----------------|------------|--------|
| CNN | 96.19% | 93.61% | 97.29% | 95.45% | 6.3M |
| DenseNet121 | **96.32%** | **93.77%** | **97.39%** | **95.58%** | 7.4M |
| Column RNN | 88.41% | 78.76% | 92.03% | 85.40% | 405K |
| MD-RNN | 88.47% | 79.34% | 92.01% | 85.67% | 874K |

CNN and DenseNet121 are the top performers (>96% accuracy). The RNN variants are more lightweight but still competitive.

## 🚀 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/cracked-surface-detection.git
cd cracked-surface-detection

# Create a virtual environment (Python 3.12 required for TensorFlow)
python3.12 -m venv tf_env
tf_env\Scripts\activate      # Windows
# or: source tf_env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Streamlit Dashboard

```bash
tf_env\Scripts\streamlit run app.py
```

Opens a browser with two tabs:
- **Detection Dashboard** — upload an image and run ensemble inference
- **Model Performance** — browse offline evaluation metrics and confusion matrices

### Training (from scratch)

```bash
# Activate venv, then run any script:
tf_env\Scripts\python src\train_cnn.py
tf_env\Scripts\python src\train_transfer.py
# etc.
```

Data is downloaded from Kaggle and split automatically on first run.

All scripts accept:
- `--epochs` — number of epochs
- `--resume` — resume from best checkpoint
- `--data-root` — custom data path (default: `data/split`)
- `--checkpoint-dir` — where to save `.keras` files (default: `models/`)
- `--results-dir` — where to save plots and CSV logs (default: `results/<model>/`)

## 📁 Project Structure

```
├── app.py                       # Streamlit dashboard
├── requirements.txt
├── config/
│   └── config.json              # Kaggle dataset link & split ratios
├── src/
│   ├── train/
│   │   ├── train_ffnn.py        # Training per model
│   │   ├── train_cnn.py
│   │   ├── train_rnn_column.py
│   │   ├── train_rnn_md.py
│   │   └── train_transfer.py
│   ├── utils/
│   │   ├── data_loader.py       # load_datasets, prefetch, class_weights, augmentation
│   │   ├── data_setup.py        # Download + split (idempotent, skips if done)
│   │   └── evaluate.py          # plot_history, confusion_matrix
│   └── models/
│       ├── ffnn.py, cnn.py, rnn_column.py, rnn_md.py, transfer.py
├── models/
│   ├── best_*.keras             # Pre-trained weights
│   └── evaluation_results.json  # Offline metrics & confusion matrices
├── notebooks/
│   └── model_architectures/     # Original Optuna hyperparameter search notebooks
└── data/
    └── split/                   # {train,val,test}/{Cracked,Non Cracked}/
```

## 🧪 Reproducibility

- All hyperparameters are the **Optuna best trials** — each model's architecture (layer count, units, dropout, learning rate) was discovered through Bayesian optimisation over 15–50 trials with early pruning via `TFKerasPruningCallback`
- `compute_class_weights()` uses deterministic file-count traversal (not dataset iteration)
- Training scripts use `seed=42` in `image_dataset_from_directory`
- Checkpointing saves the best model by `val_loss`
- Early stopping (patience 15) and ReduceLROnPlateau are enabled by default

## 🚢 Deployment

The app is ready for Streamlit Cloud. Deploy by:
1. Pushing to GitHub
2. Connecting the repo at [share.streamlit.io](https://share.streamlit.io)
3. Setting the Python version to 3.12 in `tf_env` or using the cloud's default

## 📄 License

MIT
