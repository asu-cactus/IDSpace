
# Bayesian Optimization for Scanned Image Generation

This folder contains scripts related to applying Bayesian Optimization (BO) on scanned images and generating synthetic scanned versions using optimized parameters.

## Files Overview

### 1. `Bayesian_search.py`
Applies Bayesian Optimization on scanned images.

- Defines search spaces such as:
  - Brightness
  - Contrast
  - Sharpness
  - Shadow offset
  - And more (you can add your own)

**How to Run:**
```bash
cd ..
python scanned_image/Bayesian_search.py --config scanned_image/scan_config.yaml 
```
---

### 2. `evaluate_parameters.py`
Contains the evaluation functions used during the BO process.

- You must configure:
  - Target domain data path
  - Trained model paths
  - Output paths
  - Paper texture path
- Sample files can be found in: `../data/scanned_data`

---

### 3. `scanned_image_generation.py`
Generates scanned-style images using the parameters found via BO.

**How to Run:**
```bash
cd ..
python scanned_image/scanned_image_generation.py --params_file /path/to/autotuned_parameters --<optional_parameters>
```

---

### 4. `utils.py`
Provides auxiliary functions used in both the optimization and image generation processes.
