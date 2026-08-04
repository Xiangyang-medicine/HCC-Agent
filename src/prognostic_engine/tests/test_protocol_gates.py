"""Publication-facing structural gates for Phase 3A outputs."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from prognostic_engine.bootstrap import patient_level_paired_bootstrap
from prognostic_engine.training import EXPECTED_MODELS, NestedCVTrainer


def _predictions(n_repeats=1):
    rows = []
    case_ids = [f"case_{i:03d}" for i in range(363)]
    for repeat in range(1, n_repeats + 1):
        for index, case_id in enumerate(case_ids):
            fold = index % 5 + 1
            for model_index, model in enumerate(EXPECTED_MODELS):
                risk = float(index / 363 + model_index * 0.01)
                rows.append({
                    'case_id': case_id,
                    'repeat': repeat,
                    'fold': fold,
                    'model': model,
                    'risk_score': risk,
                    'survival_probability_12m': 0.90,
                    'survival_probability_36m': 0.70,
                    'survival_probability_60m': 0.50,
                    'survival_months': float(10 + index),
                    'event': int(index % 3 == 0),
                })
    return rows


def _trainer(tmp_path, predictions):
    trainer = NestedCVTrainer.__new__(NestedCVTrainer)
    trainer.output_dir = tmp_path
    trainer.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.df = pd.DataFrame({'case_id': [f"case_{i:03d}" for i in range(363)]})
    trainer.sa_name = None  # Required by _save_results
    trainer.all_predictions = predictions
    n_repeats = max(row['repeat'] for row in predictions) if predictions else 1
    metric_record = {
        'harrell_c': 0.60,
        'uno_c': 0.60,
        'auc_12m': 0.60,
        'auc_36m': 0.60,
        'auc_60m': 0.60,
        'brier_12m': 0.20,
        'brier_36m': 0.20,
        'brier_60m': 0.20,
        'ibs': 0.20,
    }
    trainer.metrics_by_model = defaultdict(list, {
        model: [dict(metric_record) for _ in range(n_repeats * 5)]
        for model in EXPECTED_MODELS
    })
    trainer.model_failures = defaultdict(list)
    trainer.ph_results = defaultdict(list)
    trainer.integrity_results = []
    trainer._aggregate_metrics = lambda: {}
    trainer._bootstrap_comparison = lambda frame: {
        'M2_vs_M1': {'status': 'TEST_STUB'},
        'M3_vs_M1': {'status': 'TEST_STUB'},
        'M3_vs_M2': {'status': 'TEST_STUB'},
    }
    trainer._aggregate_integrity = lambda: {'status': 'TEST_STUB'}
    trainer._print_summary = lambda *args, **kwargs: None
    return trainer


def _save(trainer, n_repeats):
    folds = [
        {'repeat': repeat, 'fold': fold, 'results': {}}
        for repeat in range(1, n_repeats + 1)
        for fold in range(1, 6)
    ]
    now = datetime(2026, 7, 21)
    return trainer._save_results(folds, now, now, 0.0, expected_repeats=n_repeats)


def test_pilot_gate_requires_exact_1815_predictions(tmp_path):
    report = _save(_trainer(tmp_path, _predictions(1)), 1)
    validation = report['validation']
    assert report['status'] == 'PILOT_COMPLETED'
    assert validation['expected_per_model'] == 363
    assert validation['expected_total'] == 1815
    assert validation['all_required_gates_passed'] is True

    with (tmp_path / 'metrics_summary.json').open(encoding='utf-8') as handle:
        persisted = json.load(handle)
    assert persisted['validation']['no_nan_in_predictions'] is True
    assert isinstance(persisted['validation']['no_nan_in_predictions'], bool)


def test_formal_gate_requires_1815_per_model_and_9075_total(tmp_path):
    report = _save(_trainer(tmp_path, _predictions(5)), 5)
    validation = report['validation']
    assert report['status'] == 'COMPLETED'
    assert validation['expected_per_model'] == 1815
    assert validation['expected_total'] == 9075
    assert set(validation['per_model_counts'].values()) == {1815}


@pytest.mark.parametrize('mutation,failed_gate', [
    ('missing_model', 'all_models_completed'),
    ('duplicate', 'no_duplicate_records'),
    ('nan', 'no_nan_in_predictions'),
])
def test_pilot_gate_rejects_incomplete_outputs(tmp_path, mutation, failed_gate):
    rows = _predictions(1)
    if mutation == 'missing_model':
        rows = [row for row in rows if row['model'] != 'M5_deepsurv']
    elif mutation == 'duplicate':
        rows.append(dict(rows[0]))
    else:
        rows[0]['risk_score'] = np.nan

    report = _save(_trainer(tmp_path, rows), 1)
    assert report['status'] == 'FAILED_INCOMPLETE'
    assert report['validation'][failed_gate] is False


def _repeated_pair_predictions(n_patients=40, n_repeats=5):
    rows = []
    for repeat in range(1, n_repeats + 1):
        for index in range(n_patients):
            case_id = f"p{index:03d}"
            time = float(n_patients - index + 1)
            fold = index % 5 + 1
            rows.extend([
                {'case_id': case_id, 'repeat': repeat, 'fold': fold, 'model': 'A',
                 'risk_score': float(index), 'survival_months': time, 'event': 1},
                {'case_id': case_id, 'repeat': repeat, 'fold': fold, 'model': 'B',
                 'risk_score': float(-index), 'survival_months': time, 'event': 1},
            ])
    return pd.DataFrame(rows)


def test_bootstrap_uses_paired_patient_draw_across_repeats():
    result = patient_level_paired_bootstrap(
        _repeated_pair_predictions(),
        n_iterations=50,
        seed=7,
        comparison_pair=('A', 'B'),
    )
    assert result['status'] == 'SUCCESS'
    assert result['n_repeats'] == 5
    assert result['n_patients'] == 40
    assert result['multiplicity_preserved'] is True
    assert result['pairing_key'] == ['case_id', 'repeat', 'fold']
    assert result['mean_diff'] > 0


def test_bootstrap_rejects_pair_coverage_mismatch():
    frame = _repeated_pair_predictions()
    bad_index = frame[(frame['model'] == 'B') & (frame['repeat'] == 3)].index[0]
    frame = frame.drop(index=bad_index)
    with pytest.raises(ValueError, match='coverage mismatch'):
        patient_level_paired_bootstrap(frame, n_iterations=5, comparison_pair=('A', 'B'))


def test_actual_m5_prediction_direction_is_higher_hazard():
    torch = pytest.importorskip('torch')
    tt = pytest.importorskip('torchtuples')
    CoxPH = pytest.importorskip('pycox.models').CoxPH
    from prognostic_engine.metrics import harrell_c_index
    from prognostic_engine.models import M5DeepSurv

    net = torch.nn.Linear(1, 1, bias=False).float()
    with torch.no_grad():
        net.weight.fill_(1.0)
    model = M5DeepSurv()
    model.model = CoxPH(net, tt.optim.Adam(0.01))

    features = np.asarray([[-2.0], [0.0], [2.0]], dtype=np.float32)
    risk = model.predict_risk(features)
    assert risk.dtype == np.float32
    assert np.all(np.diff(risk) > 0)
    assert harrell_c_index([30, 20, 10], [1, 1, 1], risk) == pytest.approx(1.0)
