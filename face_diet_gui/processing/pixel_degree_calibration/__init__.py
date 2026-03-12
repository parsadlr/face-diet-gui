"""
Pixel-to-degree calibration subtool.

Derives a pixel-per-degree (PPD) mapping for the Tobii glasses camera from a
recording of a known green circular target.  The target appears as an ellipse
when off-axis, which is used to estimate anisotropic (x/y) PPD.

GUI (recommended)
-----------------
Launch the interactive mini-GUI with:

    python -m face_diet_gui.processing.pixel_degree_calibration

The GUI provides two tabs:
    Segment Targets – browse calibration frames, click to segment the target,
                      save masks and detections.json
    Fit Mapping     – compute PPD samples from masks and fit the polynomial
                      surfaces; displays RMSE metrics and heatmap previews.

CLI Workflow (alternative)
--------------------------
1. extract_frames.py       – dump every N-th frame from the calibration video
2. (manual)                – copy ~20-30 frames covering the full FOV to a folder
3. detect_target.py        – green-hue segmentation + ellipse fitting → masks +
                             detections.json
4. compute_samples.py      – compute PPD samples from ellipse geometry
5. fit_mapping.py          – fit three 2D polynomial surfaces (scalar, x, y)
                             (also auto-saves heatmap PNGs)
6. visualize_mapping.py    – re-save heatmap images of the fitted surfaces

Runtime API (mapping_utils.py)
-------------------------------
    load_pixel_degree_mapping(path)        → mapping dict
    evaluate_ppd(mapping, x, y)            → scalar PPD
    evaluate_ppd_xy(mapping, x, y)         → (ppd_x, ppd_y)
    estimate_distance_from_face_size(...)  → distance in metres
"""
