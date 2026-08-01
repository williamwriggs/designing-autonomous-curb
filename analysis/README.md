# Analysis scripts

This directory contains reproducible scripts used to validate the synthetic dataset and generate analysis outputs for the paper and dashboard.

Planned workflow:

1. `01_validate_data.py` — checks required fields, event counts, missing values, and category consistency.
2. `02_temporal_analysis.py` — summarizes events by hour, day, period, and event type.
3. `03_spatial_analysis.py` — summarizes generalized analytical polygons and adjusted-point coordinates.
4. `04_adaptive_curb_scenario.py` — reproduces the data-calibrated adaptive curb allocation scenario.

The Streamlit dashboard in `app.py` provides an interactive view of these analyses.
