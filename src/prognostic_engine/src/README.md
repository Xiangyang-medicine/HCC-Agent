# Prognostic Engine - Phase 3A Formal Training

## Environment Setup

### Recommended Environment
- Python 3.9-3.11 (Python 3.13 incompatible with scikit-survival 0.28-0.30)
- OS: Windows/Linux/macOS

### Package Versions
```
python>=3.9,<3.12  # Avoid >3.12 due to scikit-survival build issues
scikit-survival==0.28.0  # DO NOT upgrade to >=0.30.0 (breaking changes)
scikit-learn>=1.1.3,<1.5  # Ensure compatibility with scikit-survival 0.28
numpy~=1.24
pandas~=2.0
lifelines~=0.27
pycox~=0.2
torchtuples~=0.4.2
torch~=2.0
torchvision~=0.15
```

### Quick Install
```bash
pip install "scikit-survival==0.28.0" "scikit-learn<1.5,>=1.1.3" numpy pandas lifelines
pip install pycox torch torchtuples
```

## Run Phase 3A Training

### Training Command
```bash
cd F:/ACM/src/prognostic_engine/src
python -m prognostic_engine.training
```

### What it does
- Performs 5×5×5 nested CV (5 outer repeats × 5 outer folds × 5 inner folds)
- Trains 5 models: M1-M5 with hyperparameter tuning
- Computes Harrell C-index, Uno C-index, AUC(12/36/60), Brier/IBS, calibration
- Stores OOF predictions for bootstrap comparison
- Runs sensitivity analyses SA1 (n=363) and SA2 (n=361)

### Expected Output
- `experiments/phase3a/formal/oof_predictions.csv` – Out-of-fold predictions
- `experiments/phase3a/formal/metrics_summary.json` – Aggregated metrics
- `experiments/phase3a/formal/bootstrap_results/` – Paired bootstrap comparisons

## Notes
- This system currently trains on internal TCGA-LIHC data
- External validation (ICGC-LIRI-JP, GEO) remains as next step when network issues resolved
