# Supplementary Figure S3

Run with the locked project environment:

1. `python prepare_source_data.py`
2. Run each panel script.
3. `python assemble_supp_figure_s3.py`
4. `python verify_supp_figure_s3.py`

Each panel folder contains its own `source_data.csv`, plotting code, and
standalone SVG/PDF/PNG/TIFF outputs. The combined figure is exported in the
same four formats at the figure root.
