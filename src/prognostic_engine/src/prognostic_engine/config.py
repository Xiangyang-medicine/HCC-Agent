"""Configuration and constants for Phase 3A nested CV."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "modeling"
SPLITS_DIR = PROJECT_ROOT / "experiments" / "phase3a" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "phase3a" / "formal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metabolic genes (15 genes per SAP v1.1)
METABOLIC_GENES = [
    "HK2", "PKM", "LDHA", "LDHB", "GPI", "PFKL",  # Glycolysis
    "GLS", "GLUD1",  # Glutamine
    "FASN", "SCD",  # Lipogenesis
    "CA9", "VEGFA", "HIF1A",  # Hypoxia
    "MYC", "CTNNB1"  # Oncogenic
]

# CV configuration
N_OUTER_REPEATS = 5
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
OUTER_SEED = 42
INNER_SEED = 123

# Model hyperparameters per SAP v1.1
M1_CONFIG = {}  # Clinical-only CoxPH - no tuning needed

# M2/M3: CoxnetSurvivalAnalysis grid
# Per Phase 3A reset: dynamic generation based on data, avoiding zero-coefficient models
# Alpha is L1 penalty strength - higher values = more regularization = more zeros
# We use log-spaced values in a safe range to avoid degenerate models
M2_M3_ALPHA_RANGE = [0.01, 0.05, 0.1, 0.3]  # Conservative range avoiding extreme regularization
M2_M3_L1_RATIO_RANGE = [0.1, 0.3, 0.5, 0.7, 0.9]  # 0.0 not allowed, use 0.1-0.9

# M4: RSF grid
M4_N_ESTIMATORS_RANGE = [50, 100, 200]
M4_MAX_DEPTH_RANGE = [3, 5, None]
M4_MIN_SAMPLES_SPLIT_RANGE = [5, 10, 20]
M4_MIN_SAMPLES_LEAF_RANGE = [3, 5, 10]

# M5: DeepSurv grid (simplified for CPU)
M5_HIDDEN_LAYERS_RANGE = [[32], [64], [32, 16]]
M5_LR_RANGE = [0.001, 0.01]
M5_BATCH_FRAC_RANGE = [0.5, 1.0]

# Evaluation
EVALUATION_TIMES = [12, 36, 60]  # months
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 456

# Sensitivity analysis cohorts
SA1_N = 363  # Full cohort
SA2_N = 361  # age >= 18
