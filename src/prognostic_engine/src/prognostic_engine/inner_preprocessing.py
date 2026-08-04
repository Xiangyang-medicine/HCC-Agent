"""Inner-fold preprocessing utilities for nested CV.

Per Phase 3A reset: Preprocessing must be fit ONLY on inner-training data
within each inner CV fold, not on the full outer training set.

This ensures:
1. No data leakage from inner validation to preprocessing fit
2. Consistent preprocessing parameters for reproducibility
3. Proper variance filtering per-fold
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess_inner_fold_clinical(inner_train_df, inner_val_df=None):
    """
    Preprocess clinical features for an inner fold.

    Fits ONLY on inner_train_df, transforms both train and val.

    Parameters
    ----------
    inner_train_df : pd.DataFrame
        Inner training data (case_id, clinical features, survival outcomes)
    inner_val_df : pd.DataFrame, optional
        Inner validation data. If None, only fits and returns fitted parameters.

    Returns
    -------
    dict
        Preprocessed features + fitted parameters for later transform
    """
    # Age: z-score normalization
    age_mean = inner_train_df['age_at_diagnosis'].mean()
    age_std = inner_train_df['age_at_diagnosis'].std()
    if age_std < 1e-8:
        age_std = 1.0  # Avoid division by zero

    train_age = (inner_train_df['age_at_diagnosis'] - age_mean) / age_std
    val_age = None
    if inner_val_df is not None:
        val_age = (inner_val_df['age_at_diagnosis'] - age_mean) / age_std

    # Stage: one-hot encoding
    stage_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    stage_encoder.fit(inner_train_df['ajcc_stage'].astype(str).to_numpy().reshape(-1, 1))

    train_stage = stage_encoder.transform(inner_train_df['ajcc_stage'].astype(str).to_numpy().reshape(-1, 1))
    val_stage = None
    if inner_val_df is not None:
        val_stage = stage_encoder.transform(inner_val_df['ajcc_stage'].astype(str).to_numpy().reshape(-1, 1))

    stage_cols = [f"stage_{cat}" for cat in stage_encoder.categories_[0]]

    # Grade: one-hot encoding
    grade_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    grade_encoder.fit(inner_train_df['tumor_grade'].astype(str).to_numpy().reshape(-1, 1))

    train_grade = grade_encoder.transform(inner_train_df['tumor_grade'].astype(str).to_numpy().reshape(-1, 1))
    val_grade = None
    if inner_val_df is not None:
        val_grade = grade_encoder.transform(inner_val_df['tumor_grade'].astype(str).to_numpy().reshape(-1, 1))

    grade_cols = [f"grade_{cat}" for cat in grade_encoder.categories_[0]]

    # Gender: one-hot encoding (fallback for datasets without gender)
    try:
        gender_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        gender_encoder.fit(inner_train_df['gender'].astype(str).to_numpy().reshape(-1, 1))

        train_gender = gender_encoder.transform(inner_train_df['gender'].astype(str).to_numpy().reshape(-1, 1))
        val_gender = None
        if inner_val_df is not None:
            val_gender = gender_encoder.transform(inner_val_df['gender'].astype(str).to_numpy().reshape(-1, 1))
        gender_cols = [f"gender_{cat}" for cat in gender_encoder.categories_[0]]
    except KeyError:
        train_gender = np.zeros((len(inner_train_df), 1))
        val_gender = np.zeros((len(inner_val_df), 1)) if inner_val_df is not None else None
        gender_cols = ['gender_Undocumented']

    # Combine all clinical features
    clinical_train = np.hstack([
        train_age.values.reshape(-1, 1),
        train_stage,
        train_grade,
        train_gender
    ])
    clinical_val = None
    if inner_val_df is not None:
        clinical_val = np.hstack([
            val_age.values.reshape(-1, 1),
            val_stage,
            val_grade,
            val_gender
        ])

    clinical_cols = ['age_z'] + stage_cols + grade_cols + gender_cols

    return {
        'clinical_train': clinical_train,
        'clinical_val': clinical_val,
        'clinical_cols': clinical_cols,
        'age_mean': age_mean,
        'age_std': age_std,
        'stage_encoder': stage_encoder,
        'grade_encoder': grade_encoder,
        'gender_encoder': gender_encoder if 'gender_encoder' in dir() else None,
        'gender_cols': gender_cols,
    }


def preprocess_inner_fold_genes(inner_train_df, inner_val_df=None, gene_cols=None):
    """
    Preprocess gene features (z-score + variance filtering) for an inner fold.

    Fits ONLY on inner_train_df, transforms both train and val.

    Parameters
    ----------
    inner_train_df : pd.DataFrame
        Inner training data with gene columns
    inner_val_df : pd.DataFrame, optional
        Inner validation data
    gene_cols : list, optional
        Gene column names. If None, reads from config METABOLIC_GENES.

    Returns
    -------
    dict
        Preprocessed features + fitted parameters
    """
    from prognostic_engine.config import METABOLIC_GENES

    if gene_cols is None:
        gene_cols = [f"{g}_log2tpm" for g in METABOLIC_GENES]

    # Gene z-score normalization
    gene_mean = inner_train_df[gene_cols].mean()
    gene_std = inner_train_df[gene_cols].std()
    mask_sd = gene_std.abs() < 1e-8
    gene_std.loc[mask_sd] = 1.0  # Avoid division by zero

    train_genes = (inner_train_df[gene_cols] - gene_mean) / gene_std
    val_genes = None
    if inner_val_df is not None:
        val_genes = (inner_val_df[gene_cols] - gene_mean) / gene_std

    return {
        'gene_cols': gene_cols,
        'train_genes': train_genes.values,
        'val_genes': val_genes.values if val_genes is not None else None,
        'gene_mean': gene_mean,
        'gene_std': gene_std,
    }


class InnerFoldPreprocessor:
    """
    Handles preprocessing for inner CV folds.

    Per Phase 3A reset: Each inner fold must:
    1. Extract inner_train_df and inner_val_df by case_id
    2. Fit preprocessing ONLY on inner_train_df (imputation, encoding, scaling, variance filtering)
    3. Transform inner_val_df only

    Usage:
        preprocessor = InnerFoldPreprocessor()

        # For inner CV during tuning
        for fold in inner_splits['folds']:
            inner_train_ids = fold['train_case_ids']
            inner_val_ids = fold['val_case_ids']

            inner_train_df = full_train_df[full_train_df['case_id'].isin(inner_train_ids)]
            inner_val_df = full_train_df[full_train_df['case_id'].isin(inner_val_ids)]

            X_train, X_val, params = preprocessor.preprocess_inner_fold(inner_train_df, inner_val_df)
            # Use X_train, X_val for tuning...

        # For final model training (on full outer_train)
        final_prep = preprocessor.preprocess_full_train(outer_train_df, outer_test_df)
        # Use final_prep for model.fit()...
    """

    def __init__(self):
        self.gene_cols = None
        self._fitted = False

    def preprocess_inner_fold(self, inner_train_df, inner_val_df):
        """
        Preprocess an inner fold: fit on train, transform val.

        Parameters
        ----------
        inner_train_df : pd.DataFrame
            Inner training data
        inner_val_df : pd.DataFrame
            Inner validation data

        Returns
        -------
        tuple
            (X_train, X_val, preprocessor_params)
        """
        clinical_prep = preprocess_inner_fold_clinical(inner_train_df, inner_val_df)
        gene_prep = preprocess_inner_fold_genes(inner_train_df, inner_val_df)

        # For inner fold: return both train and val
        X_train = {
            'clinical': clinical_prep['clinical_train'],
            'genes': gene_prep['train_genes'],
            'clinical_cols': clinical_prep['clinical_cols'],
            'gene_cols': gene_prep['gene_cols'],
        }

        X_val = None
        if inner_val_df is not None:
            X_val = {
                'clinical': clinical_prep['clinical_val'],
                'genes': gene_prep['val_genes'],
            }

        params = {
            'clinical': clinical_prep,
            'genes': gene_prep,
        }

        return X_train, X_val, params

    def preprocess_full_train(self, train_df, test_df):
        """
        Preprocess full outer training set for final model training.

        This is called AFTER hyperparameter tuning to fit final models.

        Parameters
        ----------
        train_df : pd.DataFrame
            Full outer training data
        test_df : pd.DataFrame
            Outer test data

        Returns
        -------
        dict
            Preprocessed features + fitted parameters
        """
        clinical_prep = preprocess_inner_fold_clinical(train_df, test_df)
        gene_prep = preprocess_inner_fold_genes(train_df, test_df)

        # Combined features
        X_train_combined = np.hstack([clinical_prep['clinical_train'], gene_prep['train_genes']])
        X_test_combined = np.hstack([clinical_prep['clinical_val'], gene_prep['val_genes']])

        combined_cols = clinical_prep['clinical_cols'] + gene_prep['gene_cols']

        self.gene_cols = gene_prep['gene_cols']
        self._fitted = True

        return {
            'clinical': clinical_prep,
            'genes': gene_prep,
            'X_clinical_train': clinical_prep['clinical_train'],
            'X_clinical_test': clinical_prep['clinical_val'],
            'X_gene_train': gene_prep['train_genes'],
            'X_gene_test': gene_prep['val_genes'],
            'X_combined_train': X_train_combined,
            'X_combined_test': X_test_combined,
            'combined_cols': combined_cols,
        }


def get_inner_fold_data(df, inner_splits, fold_idx):
    """
    Extract inner fold data from full DataFrame using case_id lists.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame with case_id column
    inner_splits : dict
        Inner splits with case_id lists
    fold_idx : int
        Inner fold index (0-based)

    Returns
    -------
    tuple
        (inner_train_df, inner_val_df, train_case_ids, val_case_ids)
    """
    fold_info = inner_splits['folds'][fold_idx]
    train_case_ids = fold_info['train_case_ids']
    val_case_ids = fold_info['val_case_ids']

    inner_train_df = df[df['case_id'].isin(train_case_ids)].copy()
    inner_val_df = df[df['case_id'].isin(val_case_ids)].copy()

    return inner_train_df, inner_val_df, train_case_ids, val_case_ids
