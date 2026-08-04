#!/usr/bin/env python3
"""Verify the pre-specified DeepSurv risk-score direction.

Pycox CoxPH treats the neural-network output as log relative hazard. Therefore
larger network output must mean greater event risk. This check uses the actual
M5 prediction path with a deterministic one-weight network; it never flips a
sign in response to observed cohort performance.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent / "src"
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PACKAGE_DIR))


def verify_m5_risk_direction():
    import torch
    import torchtuples as tt
    from pycox.models import CoxPH

    from prognostic_engine.metrics import harrell_c_index
    from prognostic_engine.models import M5DeepSurv

    net = torch.nn.Linear(1, 1, bias=False).float()
    with torch.no_grad():
        net.weight.fill_(1.0)

    m5 = M5DeepSurv()
    m5.model = CoxPH(net, tt.optim.Adam(0.01))

    features = np.asarray([[-2.0], [0.0], [2.0]], dtype=np.float32)
    risks = m5.predict_risk(features)
    survival_months = np.asarray([30.0, 20.0, 10.0])
    events = np.ones(3, dtype=bool)
    c_index = harrell_c_index(survival_months, events, risks)

    passed = bool(
        risks.dtype == np.float32
        and np.all(np.diff(risks) > 0)
        and np.isclose(c_index, 1.0)
    )
    result = {
        "test": "M5_DEEPSURV_RISK_DIRECTION",
        "status": "PASS" if passed else "FAIL",
        "library_semantics": "pycox CoxPH network output is log relative hazard",
        "risk_convention": "higher risk_score means greater event hazard",
        "post_hoc_sign_flip": False,
        "features": features.ravel().tolist(),
        "risk_scores": risks.tolist(),
        "harrell_c": float(c_index),
        "dtype": str(risks.dtype),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    output = PROJECT_ROOT / "experiments" / "phase3a" / "readiness" / "M5_DIRECTION_TEST.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")
    return passed


if __name__ == "__main__":
    raise SystemExit(0 if verify_m5_risk_direction() else 1)
