# WBC_Detection_ECE664

Code companion for the ECE664 **Image Fusion Algorithms and Applications** term project:

**Morphology-Preserving and Task-Oriented Multi-Focus Fusion for a Custom Motorized Blood Smear Microscope**

This repository contains the first-round implementation used in the article. The current code focuses on the **acquisition and analysis stage** of the pipeline: Z-stack simulation/acquisition, autofocus scoring, best-focus selection, preliminary WBC detection, crop generation, optional classification, and JSON/Excel-style reporting.

## Project Scope

The broader project aims to move from **single best-focus frame selection** toward **true multi-focus fusion** for blood-smear microscopy. The implementation in this repository provides the experimental foundation for that goal:

1. Z-stack image acquisition or simulation  
2. Focus measure computation across the Z-axis  
3. Autofocus peak detection  
4. Best-focus frame extraction  
5. Preliminary WBC detection with OpenCV  
6. Cell crop generation for downstream learning  
7. Optional Swin Transformer classification  
8. Structured output reporting

## Main Script

- `wbc_image_pipeline_homework.py`  
  End-to-end academic homework script covering acquisition, autofocus, preprocessing, detection, cropping, classification, and report generation.

## Features

### 1) Camera acquisition and Z-stack workflow
The script can operate in two modes:
- **real hardware mode** using a Basler camera (`pypylon`) and serial-controlled Z-stage (`pyserial`)
- **simulation mode** when hardware is unavailable

### 2) Focus analysis
Implemented focus measures include:
- Laplacian variance
- Tenengrad
- contrast score
- entropy score
- combined autofocus score

The current combined autofocus score is:

`Combined Score = 0.70 * Laplacian + 0.30 * Tenengrad`

### 3) Preliminary WBC detection
The script uses an OpenCV-based contour pipeline with:
- CLAHE
- HSV thresholding
- morphological filtering
- bounding-box extraction

### 4) Cell cropping
Detected cells are:
- squared
- enlarged with contextual margin
- resized to `360 x 360`

### 5) Optional classification
A Swin Transformer Small (`swin_s`) classifier is included.  
If PyTorch weights are unavailable, the script falls back to **simulation mode** for classification output.

### 6) Reporting
The script exports:
- focus curve figure
- detected-cell visualization
- cropped cell images
- JSON summary report
- focus metrics spreadsheet / CSV fallback

## Expected Outputs

Typical generated outputs include:

- `captured_focus_peak.png`
- `focus_curve.png`
- `detected_cells.png`
- `crops/`
- `wbc_results_v1.json`
- `focus_measure_results.xlsx` or CSV equivalent

## Installation

Create a Python environment and install the required packages.

```bash
pip install -r requirements.txt
```

If you prefer manual installation:

```bash
pip install opencv-python numpy matplotlib pillow torch torchvision pandas openpyxl pyserial
```

Optional hardware libraries:
- `pypylon` for Basler camera integration
- `pyserial` for motor/stage communication

## Usage

Run:

```bash
python wbc_image_pipeline_homework.py
```

The script will:
- try to connect to the camera and motorized stage
- switch to simulation mode if hardware is unavailable
- acquire a Z-stack
- compute focus scores
- choose the best-focus frame
- run preliminary WBC detection
- generate crops and reports

## Hardware Notes

This project was developed for a **custom motorized blood-smear microscope** built by converting a manual microscope into a motorized scanning platform. Embedded control software, serial communication, and image acquisition logic were implemented in-house.

## Current Limitations

- The present implementation performs **best-focus selection**, not full multi-focus fusion.
- Stitching is not yet included in this repository version.
- Detection is preliminary and not yet a final clinical detector.
- Classification may run in simulation mode if model weights are missing.
- The article reports **first-round experimental results** focused mainly on autofocus/focus-measure behavior.

## Planned Next Steps

- Top-k candidate-frame retention instead of single-frame selection
- Laplacian pyramid-based multi-focus fusion
- structure-preserving clinician-view rendering
- stitching of fused fields of view
- task-oriented evaluation with WBC/RBC detection
- comparative experiments against classical and deep fusion baselines

## Repository Citation

If you use this repository in a report or article, cite it as:

> Salih Yalcin, "WBC_Detection_ECE664," GitHub repository, commit `cf9e5d19c5c9bea7ac862a08504ccd404aa841cd`, 2026.

Repository URL:

`https://github.com/salihyalcin38/AI_Projects/tree/cf9e5d19c5c9bea7ac862a08504ccd404aa841cd/WBC_Detection_ECE664`

## Relation to the Article

This repository is the public code companion of the ECE664 article and is cited in the manuscript as the implementation source for the first-round submission.

## AI Acknowledgement

ChatGPT and Antigravity were used for structuring documentation, refining wording, and organizing the article/report workflow. All technical content, code choices, and final responsibility belong to the author.
