"""Model implementations for Phase 3A.

M1: Clinical-only Cox PH (no tuning - baseline)
M2: Gene-only Coxnet (inner CV tuning)
M3: Combined Coxnet (inner CV tuning)
M4: Combined Random Survival Forest (inner CV tuning)
M5: Combined DeepSurv (inner CV tuning)
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from prognostic_engine.metrics import harrell_c_index, uno_c_index

# Shim torch if not available to allow non-M5 runs
try:
    import torch
    import torchtuples as tt
except ImportError:
    torch = None
    tt = None
    class MockTorch:
        @staticmethod
        def version():
            return "0.0.0"
        class optim:
            class Adam:
                def __init__(self, lr): pass
        class utils:
            class Sequence:
                def __init__(self, data): pass
        class nn:
            class Module:
                pass
            class Linear:
                def __init__(self, a, b): pass
            class ReLU:
                pass
            class BatchNorm1d:
                pass
            LayerNorm = type.__class__  # Allow LayerNorm fallback
            Dropout = type.__class__
            class Sequential:
                def __init__(self, *args): pass
            class Adam:
                def __init__(self, lr): pass
        class optimizers:
            pass
        def tensor(x): return x
        def zeros(*args, **kwargs): return np.zeros(*args, **kwargs)
        class float32: pass
    class MockTTS:
        class optim:
            Adam = lambda lr: None
        class callbacks:
            EarlyStopping = lambda p: None
        class models:
            class CoxPH:
                def __init__(self, net, opt): pass
                def fit(self, *a, **k): pass
                def compute_baseline_hazards(self): pass
                def predict(self, x): return np.ones(len(x)).reshape(-1, 1).astype(np.float32)
    torch = MockTorch()
    tt = MockTTS()

from prognostic_engine.config import (
    M1_CONFIG, M2_M3_ALPHA_RANGE, M2_M3_L1_RATIO_RANGE,
    M4_N_ESTIMATORS_RANGE, M4_MAX_DEPTH_RANGE,
    M4_MIN_SAMPLES_SPLIT_RANGE, M4_MIN_SAMPLES_LEAF_RANGE,
    M5_HIDDEN_LAYERS_RANGE, M5_LR_RANGE, M5_BATCH_FRAC_RANGE
)


class M1ClinicalCox:
    """M1: Clinical-only Cox PH model (no hyperparameter tuning)."""

    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(self, X_train, y_train, event_train, feature_names=None):
        """Fit M1 model."""
        self.feature_names = feature_names
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(X_train.shape[1])]

        # Remove low-variance columns that cause convergence issues
        X_clean, mask_keep = self._remove_low_variance_cols(X_train, threshold=0.001)
        self._feature_mask = mask_keep
        if feature_names:
            self.feature_names = [f for f, keep in zip(feature_names, mask_keep) if keep]
        clean_cols = self.feature_names if feature_names else None

        train_df = self._create_dataframe(X_clean, y_train, event_train, clean_cols)
        self.model = CoxPHFitter(penalizer=0.5, l1_ratio=0.1)  # Stronger regularization
        self.model.fit(train_df, duration_col='time', event_col='event')
        return self

    def _remove_low_variance_cols(self, X, threshold=0.001):
        """Remove columns with variance below threshold."""
        variances = np.var(X, axis=0)
        mask = variances > threshold
        return X[:, mask], mask

    def predict_risk(self, X_test):
        """Predict risk scores (partial hazards)."""
        if self.model is None:
            raise ValueError("Model not fitted")
        # Filter to same columns used during training
        if hasattr(self, '_feature_mask'):
            X_test_clean = X_test[:, self._feature_mask]
        else:
            X_test_clean = X_test[:, :len(self.feature_names)]
        test_df = self._create_dataframe(X_test_clean, None, None, self.feature_names)
        return self.model.predict_partial_hazard(test_df).values.flatten()

    def predict_survival(self, X_test, times):
        """Predict survival probabilities at given times."""
        if self.model is None:
            raise ValueError("Model not fitted")
        # Filter to same columns used during training
        if hasattr(self, '_feature_mask'):
            X_test_clean = X_test[:, self._feature_mask]
        else:
            X_test_clean = X_test[:, :len(self.feature_names)]
        test_df = self._create_dataframe(X_test_clean, None, None, self.feature_names)
        surv_funcs = getattr(self.model, 'predict_survival_function')(test_df)
        n_patients = len(X_test)
        probs = np.zeros((n_patients, len(times)))

        # lifelines returns: DataFrame with time points as index, patients as columns
        # Shape: (n_time_points, n_patients)
        if hasattr(surv_funcs, 'index') and hasattr(surv_funcs, 'columns'):
            # DataFrame: index=times, columns=patients
            sf_times = surv_funcs.index.values
            for i, t in enumerate(times):
                closest_idx = np.argmin(np.abs(sf_times - t))
                closest_t = sf_times[closest_idx]
                # Column values are survival probs for each patient at closest_t
                probs[:, i] = surv_funcs.loc[closest_t].values
        elif callable(surv_funcs):
            # Single callable for single patient
            for i, t in enumerate(times):
                probs[:, i] = surv_funcs(t)
        elif isinstance(surv_funcs, (list, tuple)):
            # List of callables (one per patient)
            for i, t in enumerate(times):
                for j, sf in enumerate(surv_funcs):
                    if callable(sf):
                        probs[j, i] = sf(t)
                    else:
                        probs[j, i] = float(sf)
        else:
            # Per Phase 3A reset: Remove fake survival decay formula.
            # Fail explicitly rather than providing misleading survival probabilities.
            raise ValueError(
                f"Cannot compute survival probabilities: unexpected survival function "
                f"type {type(surv_funcs)}. Expected DataFrame, callable, or list of callables."
            )

        return probs

    def _create_dataframe(self, X, y, event, feature_names):
        """Create DataFrame for lifelines."""
        df = {name: X[:, i] for i, name in enumerate(feature_names)}
        if y is not None:
            df['time'] = y
        if event is not None:
            df['event'] = event.astype(int)
        return __import__('pandas').DataFrame(df)


class M2M3Coxnet:
    """M2/M3: Coxnet model with inner CV hyperparameter tuning."""

    def __init__(self, model_name='M2'):
        self.model_name = model_name
        self.model = None
        self.best_alpha = None
        self.best_l1_ratio = None
        self.feature_names = None

    def tune(self, X_train, y_train, event_train, alpha_range=M2_M3_ALPHA_RANGE,
             l1_ratio_range=M2_M3_L1_RATIO_RANGE,
             inner_cv_splits=None):
        """
        Tune hyperparameters via inner CV.

        Per Phase 3A reset:
        - Skip alpha values that produce zero-coefficient models.
        - Handle ArithmeticError (weights too large) with adaptive alpha path.
        - Use provided inner_cv_splits instead of internal KFold.

        Parameters
        ----------
        inner_cv_splits : list of (train_idx, val_idx) tuples, optional
            Pre-generated inner CV splits. If None, generates 3-fold internally.
        """
        # Structured array for sksurv
        y_struct = np.array(
            [(bool(e), t) for e, t in zip(event_train, y_train)],
            dtype=[('event', bool), ('time', float)]
        )

        best_score = -np.inf
        best_params = (None, None)

        # Per Phase 3A reset: inner_cv_splits is REQUIRED - no internal KFold fallback
        if inner_cv_splits is None:
            raise ValueError(
                "inner_cv_splits is required for M2M3Coxnet.tune(). "
                "Generate splits via inner_splits.generate_inner_splits() or pass from outer pipeline."
            )
        cv_splits = inner_cv_splits['folds']

        # Track failed alphas to build adaptive path
        failed_alphas = set()

        for alpha in alpha_range:
            # Skip alphas that previously caused ArithmeticError
            if alpha in failed_alphas:
                continue

            for l1_ratio in l1_ratio_range:
                if l1_ratio <= 0 or l1_ratio > 1:
                    continue

                scores = []

                for fold in cv_splits:
                    train_idx = fold['train_indices']
                    val_idx = fold['val_indices']
                    X_t, X_v = X_train[train_idx], X_train[val_idx]
                    y_t, y_v = y_train[train_idx], y_train[val_idx]
                    e_t, e_v = event_train[train_idx], event_train[val_idx]

                    y_t_struct = np.array(
                        [(bool(e), t) for e, t in zip(e_t, y_t)],
                        dtype=[('event', bool), ('time', float)]
                    )
                    y_v_struct = np.array(
                        [(bool(e), t) for e, t in zip(e_v, y_v)],
                        dtype=[('event', bool), ('time', float)]
                    )

                    try:
                        model = CoxnetSurvivalAnalysis(
                            alphas=[alpha],
                            l1_ratio=l1_ratio,
                            max_iter=100000
                        )
                        model.fit(X_t, y_t_struct)
                    except ArithmeticError as ae:
                        # Per Phase 3A reset: Catch "weights are too large" error
                        # Record this alpha as failed and continue to next
                        failed_alphas.add(alpha)
                        import logging
                        logging.warning(
                            f"Coxnet ArithmeticError for alpha={alpha}: {ae}. "
                            f"Marking alpha={alpha} as failed, will retry with larger alpha."
                        )
                        break  # Break inner CV loop for this alpha
                    except Exception:
                        scores.append(0.45)  # Treat other exceptions as worst score
                        continue

                    risk_pred = model.predict(X_v)

                    # Phase 3A Reset: Check for zero-coefficient model
                    coef = model.coef_
                    if coef is None or np.abs(coef).sum() < 1e-6:
                        scores.append(0.45)
                        continue

                    # Handle c-index computation errors (e.g., NoComparablePairException)
                    try:
                        cidx, _, _, _, _ = concordance_index_censored(
                            y_v_struct['event'], y_v_struct['time'], risk_pred
                        )
                        scores.append(cidx)
                    except Exception:
                        scores.append(0.45)
                        continue

                # If alpha failed (ArithmeticError), skip to next alpha
                if alpha in failed_alphas:
                    continue

                if not scores:
                    continue

                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = (alpha, l1_ratio)

        # Final validation: ensure best model has non-zero coefficients
        if best_params[0] is not None:
            try:
                final_model = CoxnetSurvivalAnalysis(
                    alphas=[best_params[0]],
                    l1_ratio=best_params[1],
                    max_iter=100000
                )
                final_model.fit(X_train, y_struct)
                coef = final_model.coef_
                if coef is None or np.abs(coef).sum() < 1e-6:
                    import logging
                    logging.warning(
                        f"Best Coxnet model has zero coefficients (alpha={best_params[0]}, "
                        f"l1_ratio={best_params[1]}). Falling back to alpha=0.1."
                    )
                    best_params = (0.1, best_params[1])
            except ArithmeticError:
                import logging
                logging.warning(
                    f"Best Coxnet alpha={best_params[0]} failed on final fit. "
                    f"Falling back to alpha=0.1."
                )
                best_params = (0.1, best_params[1])

        self.best_alpha, self.best_l1_ratio = best_params
        return best_params

    def fit(self, X_train, y_train, event_train, feature_names=None, alpha=None, l1_ratio=None):
        """Fit with specified or tuned hyperparameters.

        Per Phase 3A reset: use fit_baseline_model=True for survival function estimation.
        """
        self.feature_names = feature_names
        if alpha is None:
            alpha = self.best_alpha
        if l1_ratio is None:
            l1_ratio = self.best_l1_ratio

        # Structured array
        y_struct = np.array(
            [(bool(e), t) for e, t in zip(event_train, y_train)],
            dtype=[('event', bool), ('time', float)]
        )

        # Phase 3A Reset: Use fit_baseline_model=True to enable survival function prediction
        self.model = CoxnetSurvivalAnalysis(
            alphas=[alpha],
            l1_ratio=l1_ratio,
            max_iter=100000,
            fit_baseline_model=True  # CRITICAL: enables baseline survival prediction
        )
        self.model.fit(X_train, y_struct)
        return self

    def predict_risk(self, X_test):
        """Predict risk scores."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X_test)

    def predict_survival(self, X_test, times):
        """Predict survival probabilities using baseline survival function.

        Per Phase 3A reset: Use baseline model survival prediction.
        Use the actual trained alpha, not hardcoded 0.05.
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        # Get the actual alpha used during training
        trained_alpha = self.best_alpha if self.best_alpha is not None else 0.1

        # Get survival functions from the model
        # Remove hardcoded alpha=0.05 - use trained alpha
        surv_functions = getattr(self.model, 'predict_survival_function')(X_test)
        n_patients = len(X_test)
        probs = np.zeros((n_patients, len(times)))

        # surv_functions: array of (time, survival) tuples or callable
        for j in range(n_patients):
            sf = surv_functions[j]
            for i, t in enumerate(times):
                if callable(sf):
                    probs[j, i] = sf(t)
                else:
                    # Array-like: find closest time point
                    probs[j, i] = np.interp(t, sf[0], sf[1])

        return probs


class M4RSF:
    """M4: Random Survival Forest with inner CV tuning."""

    def __init__(self):
        self.model = None
        self.best_params = None

    def tune(self, X_train, y_train, event_train, inner_cv_splits=None):
        """Tune hyperparameters via inner CV (simplified grid).

        Per Phase 3A reset: Use provided inner_cv_splits instead of internal KFold.
        """
        y_struct = np.array(
            [(bool(e), t) for e, t in zip(event_train, y_train)],
            dtype=[('event', bool), ('time', float)]
        )

        best_score = -np.inf
        best_params = {}

        # Simplified grid for speed
        configs = [
            {'n_estimators': 50, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
            {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 10, 'min_samples_leaf': 5},
            {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 3},
        ]

        # Per Phase 3A reset: inner_cv_splits is REQUIRED - no internal KFold fallback
        if inner_cv_splits is None:
            raise ValueError(
                "inner_cv_splits is required for M4RSF.tune(). "
                "Generate splits via inner_splits.generate_inner_splits() or pass from outer pipeline."
            )
        cv_splits = inner_cv_splits['folds']

        for config in configs:
            scores = []
            for fold in cv_splits:
                train_idx = fold['train_indices']
                val_idx = fold['val_indices']
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t_struct = y_struct[train_idx]
                y_v_struct = y_struct[val_idx]

                model = RandomSurvivalForest(**config, random_state=42, n_jobs=-1)
                model.fit(X_t, y_t_struct)
                risk_pred = model.predict(X_v)

                # Handle c-index computation errors (e.g., NoComparablePairException)
                try:
                    cidx, _, _, _, _ = concordance_index_censored(
                        y_v_struct['event'], y_v_struct['time'], risk_pred
                    )
                    scores.append(cidx)
                except Exception:
                    scores.append(0.5)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = config

        self.best_params = best_params
        return best_params

    def fit(self, X_train, y_train, event_train, **params):
        """Fit RSF with specified or tuned parameters."""
        if self.best_params is not None and not params:
            params = self.best_params

        y_struct = np.array(
            [(bool(e), t) for e, t in zip(event_train, y_train)],
            dtype=[('event', bool), ('time', float)]
        )

        self.model = RandomSurvivalForest(**params, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_struct)
        return self

    def predict_risk(self, X_test):
        """Predict risk scores."""
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X_test)

    def predict_survival(self, X_test, times):
        """Predict survival probabilities.

        Per Phase 3A reset: Use proper baseline survival extraction.
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        n_patients = len(X_test)
        probs = np.zeros((n_patients, len(times)))

        # Get all survival functions once (returns array of callables)
        surv_functions = getattr(self.model, 'predict_survival_function')(X_test)

        for j in range(n_patients):
            sf = surv_functions[j]
            for i, t in enumerate(times):
                probs[j, i] = sf(t)

        return probs


class M5DeepSurv:
    """M5: DeepSurv neural network with inner CV tuning."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_iter = None
        self.feature_names = None

    def tune(self, X_train, y_train, event_train, batch_frac=0.5, inner_cv_splits=None):
        """Tune hyperparameters via inner CV (simplified).

        Per Phase 3A reset: Use provided inner_cv_splits instead of internal KFold.
        """
        configs = [
            {'hidden_layers': [32], 'lr': 0.01, 'batch_frac': 1.0},  # Use full batch
            {'hidden_layers': [64], 'lr': 0.01, 'batch_frac': 1.0},
            {'hidden_layers': [32, 16], 'lr': 0.001, 'batch_frac': 1.0},
        ]

        # Override batch_frac for tuning phase
        for config in configs:
            config['batch_frac'] = batch_frac

        best_score = -np.inf
        best_params = {}

        # Per Phase 3A reset: inner_cv_splits is REQUIRED - no internal KFold fallback
        if inner_cv_splits is None:
            raise ValueError(
                "inner_cv_splits is required for M5DeepSurv.tune(). "
                "Generate splits via inner_splits.generate_inner_splits() or pass from outer pipeline."
            )
        cv_splits = inner_cv_splits['folds']

        for config in configs:
            scores = []
            for fold in cv_splits:
                train_idx = fold['train_indices']
                val_idx = fold['val_indices']
                X_t, X_v = X_train[train_idx], X_train[val_idx]
                y_t, y_v = y_train[train_idx], y_train[val_idx]
                e_t, e_v = event_train[train_idx], event_train[val_idx]

                try:
                    model, _ = self._train_deepsurv(
                        X_t, y_t, e_t,
                        hidden_layers=config['hidden_layers'],
                        lr=config['lr'],
                        batch_frac=config['batch_frac'],
                        epochs=50,
                        verbose=False
                    )
                    risk_pred = model.predict(np.asarray(X_v, dtype=np.float32)).flatten()

                    y_v_struct = np.array(
                        [(bool(e), t) for e, t in zip(e_v, y_v)],
                        dtype=[('event', bool), ('time', float)]
                    )
                    cidx, _, _, _, _ = concordance_index_censored(
                        y_v_struct['event'], y_v_struct['time'], risk_pred
                    )
                    scores.append(cidx)
                except Exception:
                    scores.append(0.5)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = config

        self.best_params = best_params
        return best_params

    def fit(self, X_train, y_train, event_train, **params):
        """Fit DeepSurv with specified or tuned parameters."""
        if self.best_params is not None and not params:
            params = self.best_params

        # The network must output a scalar (risk score)
        self.model, _ = self._train_deepsurv(
            X_train, y_train, event_train,
            hidden_layers=params.get('hidden_layers', [32]),
            lr=params.get('lr', 0.01),
            batch_frac=params.get('batch_frac', 0.5),
            epochs=100,
            verbose=True
        )
        return self

    def predict_risk(self, X_test):
        """Predict log-relative hazard; larger values mean greater event risk."""
        if self.model is None:
            raise ValueError("Model not fitted")
        X_test = np.array(X_test, dtype=np.float32)
        return self.model.predict(X_test).flatten()

    def predict_survival(self, X_test, times):
        """Predict survival probabilities.

        Per Phase 3A reset: Use compute_baseline_hazards() + predict_surv_df()
        to get survival curves indexed by real survival times, not linear indices.
        pycox CoxPH.predict_surv_df returns DataFrame indexed by real times
        with patients as columns.
        """
        if self.model is None:
            raise ValueError("Model not fitted")

        X_test = np.array(X_test, dtype=np.float32)
        n_patients = len(X_test)
        n_times = len(times)
        probs = np.zeros((n_patients, n_times))

        # Per Phase 3A reset: compute baseline hazards, then predict_surv_df
        # Returns DataFrame: index = real survival times, columns = patients
        self.model.compute_baseline_hazards()
        surv_df = self.model.predict_surv_df(X_test)

        if hasattr(surv_df, 'index') and hasattr(surv_df, 'columns'):
            grid_times = surv_df.index.values.astype(float)
            for j in range(n_patients):
                patient_surv = surv_df.iloc[:, j].values  # S(t) for this patient
                for i, t in enumerate(times):
                    if t <= grid_times[0]:
                        probs[j, i] = patient_surv[0]
                    elif t >= grid_times[-1]:
                        probs[j, i] = patient_surv[-1]
                    else:
                        probs[j, i] = np.interp(t, grid_times, patient_surv)
        else:
            raise ValueError(
                f"Cannot extract survival probabilities: unexpected type {type(surv_df)}. "
                f"Expected DataFrame with time index and patient columns."
            )

        return probs

    def _train_deepsurv(self, X, y, event, hidden_layers, lr, batch_frac, epochs, verbose):
        """Train DeepSurv model."""
        import torch
        import torch.nn as nn
        from pycox.models import CoxPH

        class DeepSurvNet(nn.Module):
            """DeepSurv network - outputs scalar risk score."""
            def __init__(self, in_features, hidden_layers, dropout=0.1):
                super().__init__()
                layers = []
                prev_features = in_features
                for h in hidden_layers:
                    layers.append(nn.Linear(prev_features, h))
                    # Use LayerNorm instead of BatchNorm1d to handle single-sample batches
                    layers.append(nn.LayerNorm(h))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    prev_features = h
                self.hidden = nn.Sequential(*layers)
                self.risk = nn.Linear(prev_features, 1)  # Final layer outputs scalar

            def forward(self, x):
                x = self.hidden(x)
                return self.risk(x)

        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        event = np.ascontiguousarray(event, dtype=np.float32)

        torch.manual_seed(456)

        in_features = X.shape[1]
        batch_size = max(16, int(len(X) * batch_frac))

        net = DeepSurvNet(
            in_features,
            hidden_layers,
            dropout=0.1
        ).float()

        first_parameter = next(net.parameters())
        if X.dtype != np.float32 or y.dtype != np.float32 or event.dtype != np.float32:
            raise TypeError("DeepSurv inputs must all use float32")
        if first_parameter.dtype != torch.float32:
            raise TypeError("DeepSurv network parameters must use torch.float32")

        model = CoxPH(net, tt.optim.Adam(lr))

        # No validation data to avoid shape issues
        callbacks = [tt.callbacks.EarlyStopping(patience=10)]

        _ = model.fit(
            X, (y, event),
            batch_size,
            epochs,
            callbacks,
            verbose=verbose
        )

        _ = model.compute_baseline_hazards()

        return model, None
