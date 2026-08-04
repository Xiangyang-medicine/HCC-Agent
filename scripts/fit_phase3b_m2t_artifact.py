#!/usr/bin/env python3
"""Freeze the TCGA-only M2T microarray-transport artifact."""

from pathlib import Path

from prognostic_engine.microarray_transport import fit_frozen_m2t_artifact


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    artifact, manifest = fit_frozen_m2t_artifact(
        ROOT / "data" / "modeling" / "tcga_lihc_modeling_dataset.parquet",
        ROOT / "experiments" / "phase3b" / "derivation",
    )
    print(f"artifact={artifact}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
