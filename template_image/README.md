# Bayesian Optimization for Template Image Generation

This repository contains scripts for applying **Bayesian Optimization (BO)** to template images and generating synthetic images using optimized parameters.

## File Overview
### `Optuna_search_global.py`
Performs Bayesian Optimization on template images based on Optuna library.

**Search space includes:**
- Segment position
- Font style
- Font size
- Font color
- And more (custom parameters can be added)

#### 🚀 How to Run
From the root directory of the project:
```bash
python template_image/Optuna_search_global.py \
  --config </path/to/configure.yaml> \
```
---

### `template_image_generation.py`
Generates synthetic template images using parameters found via BO.  
You must configure the template layout, portrait image, and PII fields.

#### 🚀 How to Run
```bash
python template_image_generation.py \
  --best_setting_path </path/to/best/setting.json> # usually from the BO search output\
```

---

### `utils.py`
Provides shared utility functions for both the optimization and image generation workflows.


