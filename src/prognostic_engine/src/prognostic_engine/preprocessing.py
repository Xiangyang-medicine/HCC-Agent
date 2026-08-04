"""Data preprocessing utilities for nested CV."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from prognostic_engine.config import METABOLIC_GENES


def normalize_stage(val):
    """Map AJCC stage to ordinal - deprecated, use one-hot instead."""
    if pd.isna(val):
        return "Unknown"
    return val  # Preserve original for one-hot encoding


def normalize_grade(val):
    """Map grade to ordinal - deprecated, use one-hot instead."""
    if pd.isna(val):
        return "Unknown"
    return val  # Preserve original for one-hot encoding


def one_hot_encode_stage(train_df, test_df, stage_col='ajcc_stage'):
    """One-hot encode categorical stage."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Combine train and test for consistent categories - convert to regular numpy array
    all_stage = pd.concat([train_df[stage_col].astype(str), test_df[stage_col].astype(str)]).to_numpy()
    encoder.fit(all_stage.reshape(-1, 1))

    train_stage = encoder.transform(train_df[stage_col].astype(str).to_numpy().reshape(-1, 1))
    test_stage = encoder.transform(test_df[stage_col].astype(str).to_numpy().reshape(-1, 1))

    stage_cols = [f"stage_{cat}" for cat in encoder.categories_[0]]
    return train_stage, test_stage, stage_cols


def one_hot_encode_grade(train_df, test_df, grade_col='tumor_grade'):
    """One-hot encode categorical grade."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    all_grade = pd.concat([train_df[grade_col].astype(str), test_df[grade_col].astype(str)]).to_numpy()
    encoder.fit(all_grade.reshape(-1, 1))

    train_grade = encoder.transform(train_df[grade_col].astype(str).to_numpy().reshape(-1, 1))
    test_grade = encoder.transform(test_df[grade_col].astype(str).to_numpy().reshape(-1, 1))

    grade_cols = [f"grade_{cat}" for cat in encoder.categories_[0]]
    return train_grade, test_grade, grade_cols


def preprocess_fold_clinical(train_df, test_df):
    """Preprocess clinical features for a fold using one-hot encoding."""
    # Age: z-score normalization
    age_mean = train_df['age_at_diagnosis'].mean()
    age_std = train_df['age_at_diagnosis'].std()
    if age_std < 1e-8:
        age_std = 1.0  # Avoid division by zero
    train_age = (train_df['age_at_diagnosis'] - age_mean) / age_std
    test_age = (test_df['age_at_diagnosis'] - age_mean) / age_std

    # Stage: one-hot encoding
    train_stage, test_stage, stage_cols = one_hot_encode_stage(train_df, test_df)

    # Grade: one-hot encoding
    train_grade, test_grade, grade_cols = one_hot_encode_grade(train_df, test_df)

    # Gender: one-hot encoding (per SAP v1.1 Section X, even though some datasets may have issues)
    # After checking PHASE3A_PRETRAINING_AMENDMENT.md, gender appears to be missing in TCGA-LIHC
    # We'll still include the column structure for consistency with specification
    try:
        train_gender, test_gender, gender_cols = one_hot_encode_stage(train_df, test_df, stage_col='gender')
    except KeyError:
        # Fallback for datasets without gender column
        train_gender, test_gender = np.zeros((len(train_df), 1)), np.zeros((len(test_df), 1))
        gender_cols = ['gender_Undocumented']

    # Combine all clinical features
    n_train = len(train_df)
    n_test = len(test_df)
    clinical_train = np.hstack([
        train_age.values.reshape(-1, 1),
        train_stage,
        train_grade,
        train_gender
    ])
    clinical_test = np.hstack([
        test_age.values.reshape(-1, 1),
        test_stage,
        test_grade,
        test_gender
    ])

    clinical_cols = ['age_z'] + stage_cols + grade_cols + gender_cols

    return {
        'train_age': train_age.values,
        'test_age': test_age.values,
        'train_stage': train_stage,
        'test_stage': test_stage,
        'stage_cols': stage_cols,
        'train_grade': train_grade,
        'test_grade': test_grade,
        'grade_cols': grade_cols,
        'train_gender': train_gender,
        'test_gender': test_gender,
        'gender_cols': gender_cols,
        'clinical_train': clinical_train,
        'clinical_test': clinical_test,
        'clinical_cols': clinical_cols
    }


def preprocess_fold_genes(train_df, test_df):
    """Preprocess gene features (z-score normalization)."""
    gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

    # Gene z-score normalization
    gene_mean = train_df[gene_cols].mean()
    gene_std = train_df[gene_cols].std()
    mask_sd = gene_std.abs() < 1e-8
    gene_std.loc[mask_sd] = 1.0  # Avoid division by zero

    train_genes = (train_df[gene_cols] - gene_mean) / gene_std
    test_genes = (test_df[gene_cols] - gene_mean) / gene_std

    return {
        'gene_cols': gene_cols,
        'train_genes': train_genes.values,
        'test_genes': test_genes.values
    }
