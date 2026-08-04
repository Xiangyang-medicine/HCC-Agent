# Figure 3 quality-assurance report

## Status

- Figure-specific verification: **PASS (48/48 checks)**
- Submission preflight: **PASS (14/14 checks; 0 warnings, 0 failures)**
- Vector outputs: SVG and PDF
- Raster outputs: PNG at 300 dpi and TIFF at 600 dpi
- Final canvas: 184.1 × 148.6 mm
- TIFF dimensions: 4,350 × 3,510 pixels
- SVG text remains editable

## Typography and layout

- Figure 3 uses the same two-tier heading system as Figure 2: a concise
  7.6-pt result-oriented title and a 5.35-pt muted methodological subtitle.
- Panel letters, title baselines and left margins are aligned across panels.
- Model-scope language is explicit in the panel headings: the transported
  component is gene-only and is not presented as external validation of M4.
- The top margin prevents title or panel-label clipping at the final
  double-column size.

## Data and analysis boundaries

- The displayed model is the frozen M2T gene-only elastic-net Cox transport model.
- External outcomes were not used for probe mapping, scaling, prognostic-model
  fitting, recalibration, model selection or threshold selection.
- Panel b uses external outcomes only to estimate the evaluation association
  between the prespecified continuous score and OS; this is not model recalibration.
- GSE14520/GPL3921 and GSE116174/GPL570 were evaluated separately;
  patient-level data were not pooled.
- Confidence intervals for concordance were obtained from 1,000 valid
  patient-level bootstrap iterations per estimate.
- GSE14520/GPL571 (N=21) was not analysed because of insufficient precision.
- The figure supports exploratory cross-platform transportability, not clinical
  utility or independent validation of the M4 model.

## Integrity checks

- Frozen model artifact hash matched the registered manifest.
- The frozen risk-group threshold was reproduced from the locked TCGA derivation
  data, whose SHA-256 hash matched the model manifest.
- All 285 external patients were assigned using that single TCGA-derived
  threshold without opening external outcomes.
- The previous sample-flow panel was removed from the main figure and retained
  only as supplementary source data.
- Exactly 15 candidate coefficients were loaded; PKM and LDHA were the two
  non-zero coefficients.
- Two continuous-score Cox associations, confidence intervals and
  proportional-hazards diagnostic P values were reproduced from patient-level
  evaluation records.
- Four external discrimination estimates and their confidence intervals were
  reproduced from source data.
- All displayed values were finite, and each confidence interval contained its
  point estimate.
- All panel source data, plotting code, legends, hashes and standalone panel
  exports are stored within the Figure 3 directory.
