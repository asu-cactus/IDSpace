#!/usr/bin/env python
# coding=utf-8
#!/usr/bin/env python
# coding=utf-8

import os
import sys
import json
import math
import random
import argparse

# Optional: install PyYAML if not present
try:
    import yaml
except ImportError:
    yaml = None

from utils import *  # expects simulate_scan(...)

# -------------------------
# Mapping & casting helpers
# -------------------------

INT_PARAM_KEYS = {
    # integer-like best-params that need rounding
    "id_resized_shape1", "id_resized_shape2",
    "shadow_offset1", "shadow_offset2",
    "shadow_blur_radius", "shadow_color",
    "top_left1", "top_left2",
    "top_right1", "top_right2",
    "bottom_left1", "bottom_left2",
    "bottom_right1", "bottom_right2",
    "save_quality1", "save_quality2",
}

def _as_int(val, default=None):
    try:
        return int(round(float(val)))
    except Exception:
        return default

def _as_float(val, default=None):
    try:
        return float(val)
    except Exception:
        return default

def load_saved_params(path):
    """
    Load parameters from a YAML or JSON file.

    Supported shapes:
      YAML/JSON with:
        best:
          params_eval_casted: { ... }        # preferred
          # or
          params_raw_float: { ... }          # will be casted
    If neither is found, we try top-level dict as params map.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Params file not found: {path}")

    ext = os.path.splitext(path.lower())[1]
    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML files. Install with `pip install pyyaml`.")
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    else:
        with open(path, "r") as f:
            data = json.load(f)

    # Try common result schema
    if isinstance(data, dict) and "best" in data and isinstance(data["best"], dict):
        best = data["best"]
        if "params_eval_casted" in best and isinstance(best["params_eval_casted"], dict):
            return best["params_eval_casted"]
        if "params_raw_float" in best and isinstance(best["params_raw_float"], dict):
            # cast ints where appropriate
            raw = best["params_raw_float"]
            out = {}
            for k, v in raw.items():
                if k in INT_PARAM_KEYS:
                    out[k] = _as_int(v)
                else:
                    out[k] = _as_float(v)
            return out

    # Fallback: assume whole file is the param map
    if isinstance(data, dict):
        return data

    raise ValueError(f"Unrecognized parameter file structure in: {path}")


def map_to_scan_args(saved: dict, defaults: argparse.Namespace) -> dict:
    """
    Map saved BO parameter names to simulate_scan expected keys,
    falling back to defaults for anything missing.

    Expected saved keys (from your BO script):
      brightness, contrast, sharpness_factor, noise_std, blur_radius,
      shadow_offset1, shadow_offset2, shadow_blur_radius, shadow_color,
      id_resized_shape1, id_resized_shape2,
      top_left1, top_left2, top_right1, top_right2,
      bottom_left1, bottom_left2, bottom_right1, bottom_right2,
      save_quality1, save_quality2

    simulate_scan expects:
      brightness, contrast, sharpness_factor, noise_std, blur_radius,
      shadow_offset (tuple), shadow_blur_radius, rotate,
      position1, position2, save_quality,
      id_resized_shape (tuple),
      shadow_color (tuple RGBA),
      top_left/top_right/bottom_left/bottom_right (tuples)
    """

    # Scalars
    brightness = _as_float(saved.get("brightness"), defaults.brightness)
    contrast = _as_float(saved.get("contrast"), defaults.contrast)
    sharpness_factor = _as_float(saved.get("sharpness_factor"), defaults.sharpness_factor)
    noise_std = _as_float(saved.get("noise_std"), defaults.noise_std)
    blur_radius = _as_float(saved.get("blur_radius"), defaults.blur_radius)
    shadow_blur_radius = _as_int(saved.get("shadow_blur_radius"), _as_int(defaults.shadow_blur_radius))

    # Offsets -> tuples
    sx = _as_int(saved.get("shadow_offset1"), None)
    sy = _as_int(saved.get("shadow_offset2"), None)
    shadow_offset = (sx, sy) if sx is not None and sy is not None else defaults.shadow_offset

    # color
    # Your BO uses a single 'shadow_color' (0..128). Map to RGBA: (0, 0, 0, alpha)
    shadow_alpha = _as_int(saved.get("shadow_color"), None)
    if shadow_alpha is None:
        shadow_color = defaults.shadow_color
    else:
        shadow_color = (0, 0, 0, max(0, min(255, shadow_alpha)))

    # size tuple
    w = _as_int(saved.get("id_resized_shape1"), None)
    h = _as_int(saved.get("id_resized_shape2"), None)
    id_resized_shape = (w, h) if w is not None and h is not None else defaults.id_resized_shape

    # perspective corners
    def _pair(a, b, fallback):
        ai = _as_int(saved.get(a), None)
        bi = _as_int(saved.get(b), None)
        return (ai, bi) if (ai is not None and bi is not None) else fallback

    top_left = _pair("top_left1", "top_left2", defaults.top_left)
    top_right = _pair("top_right1", "top_right2", defaults.top_right)
    bottom_left = _pair("bottom_left1", "bottom_left2", defaults.bottom_left)
    bottom_right = _pair("bottom_right1", "bottom_right2", defaults.bottom_right)

    # save quality: choose save_quality1 if present, else save_quality2, else default
    sq1 = saved.get("save_quality1")
    sq2 = saved.get("save_quality2")
    save_quality = _as_int(sq1, None)
    if save_quality is None:
        save_quality = _as_int(sq2, defaults.save_quality)
    save_quality = max(1, min(100, save_quality))

    # Not part of BO → keep from CLI/defaults (or randomize later)
    rotate = defaults.rotate
    position1 = defaults.position1
    position2 = defaults.position2

    mapped = {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness_factor": sharpness_factor,
        "noise_std": noise_std,
        "blur_radius": blur_radius,
        "shadow_offset": shadow_offset,
        "shadow_blur_radius": shadow_blur_radius,
        "rotate": rotate,
        "position1": position1,
        "position2": position2,
        "save_quality": save_quality,
        "id_resized_shape": id_resized_shape,
        "shadow_color": shadow_color,
        "top_left": top_left,
        "top_right": top_right,
        "bottom_left": bottom_left,
        "bottom_right": bottom_right,
    }
    return mapped


def save_image(
    img,
    output_png,
    dpi=300,
    color_mode="color",
    assumed_dpi_if_missing=300
):
    #img = Image.open(input_png)

    # --- Resample to simulate DPI ---
    in_dpi = img.info.get("dpi", (assumed_dpi_if_missing,))[0]
    scale = dpi / float(in_dpi)

    if scale != 1.0:
        w, h = img.size
        img = img.resize(
            (round(w * scale), round(h * scale)),
            Image.Resampling.LANCZOS
        )

    # --- Color mode conversion ---
    cm = color_mode.lower()
    if cm == "color":
        img = img.convert("RGB")
    elif cm == "grayscale":
        img = img.convert("L")
    elif cm in {"black and white", "black_and_white", "bw"}:
        gray = img.convert("L")
        img = gray.point(lambda p: 255 if p >= 128 else 0, mode="1")
    else:
        raise ValueError("color_mode must be color, grayscale, or black and white")

    # --- Save with DPI metadata ---
    img.save(output_png, dpi=(dpi, dpi))



# -------------
# Main program
# -------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scanned ID card processing using saved parameters")

    # Path args / I/O
    parser.add_argument('--params_file', type=str, default='./data/outputs/ALB/scanned_best_settings.yaml',
                        help='YAML/JSON file containing saved best parameters (e.g., best_settings.yaml)')
    parser.add_argument('--input_file_path', type=str, default='./data/inputs/ALB/scanned/sample.png', help='Input ID path')
    parser.add_argument('--output_image_folder', type=str, default='./data/outputs/ALB/scanned', help='Output image folder')
    parser.add_argument('--output_info_folder', type=str, default='./data/outputs/ALB/scanned', help='Output info folder')
    parser.add_argument('--paper_texture_path', type=str, default='./data/inputs/papers/1.png', help='Paper texture path')
    parser.add_argument('--prefix_name', type=str, default='scanned', help='Prefix name for outputs')

    # Defaults for fields not part of BO (or used when missing from params file)
    parser.add_argument('--rotate', type=float, default=random.uniform(-90, 90), help='Rotate angle for ID card')
    parser.add_argument('--position1', type=int, default=int(random.uniform(600, 1300)), help='Horizontal position')
    parser.add_argument('--position2', type=int, default=int(random.uniform(600, 1300)), help='Vertical position')
    parser.add_argument('--dpi', type=int, default=200, help='target dpi')
    parser.add_argument('--assumed_dpi_if_missing', type=int, default=300, help='Original default dpi')
    parser.add_argument('--color_mode', type=str, default='bw', help='choose from [color, grayscale, black_and_white]')

    # Reasonable fallbacks if params file is incomplete
    parser.add_argument('--brightness', type=float, default=1.0, help='Fallback brightness')
    parser.add_argument('--contrast', type=float, default=1.0, help='Fallback contrast')
    parser.add_argument('--sharpness_factor', type=float, default=1.0, help='Fallback sharpness factor')
    parser.add_argument('--noise_std', type=float, default=1.0, help='Fallback noise std')
    parser.add_argument('--blur_radius', type=float, default=0.5, help='Fallback blur radius')
    parser.add_argument('--shadow_offset', type=tuple, default=(0, 0), help='Fallback shadow shift')
    parser.add_argument('--shadow_blur_radius', type=int, default=2, help='Fallback shadow blur radius')
    #parser.add_argument('--save_quality', type=int, default=85, help='Fallback JPEG quality (1..100)')
    parser.add_argument('--id_resized_shape', type=tuple, default=(1013, 641), help='Fallback resized ID shape')
    parser.add_argument('--shadow_color', type=tuple, default=(0, 0, 0, 128), help='Fallback shadow color RGBA')
    parser.add_argument('--top_left', type=tuple, default=(60, 60), help='Fallback TL corner offset')
    parser.add_argument('--top_right', type=tuple, default=(60, 60), help='Fallback TR corner offset')
    parser.add_argument('--bottom_left', type=tuple, default=(60, 60), help='Fallback BL corner offset')
    parser.add_argument('--bottom_right', type=tuple, default=(60, 60), help='Fallback BR corner offset')

    args = parser.parse_args()

    # Compose filenames
    paper_name = os.path.splitext(os.path.basename(args.paper_texture_path))[0]
    output_name = f"{args.prefix_name}_{paper_name}_{os.path.splitext(os.path.basename(args.input_file_path))[0]}"
    file_name = output_name + '.jpg'
    json_name = output_name + '.json'
    output_image_path = os.path.join(args.output_image_folder, file_name)
    output_json_path = os.path.join(args.output_info_folder, json_name)

    os.makedirs(args.output_image_folder, exist_ok=True)
    os.makedirs(args.output_info_folder, exist_ok=True)

    # Load saved params and map them to simulate_scan() inputs
    saved_params = load_saved_params(args.params_file)
    bps_dict = map_to_scan_args(saved_params, args)

    # Keep some context in the saved JSON
    bps_dict.update({
        "input_file_path": args.input_file_path,
        "paper_texture_path": args.paper_texture_path,
        "file_name": file_name,
        "json_name": json_name,
        "params_source": args.params_file,
        "color_mode": args.color_mode,
        "dpi": args.dpi,
    })
    bps_dict.pop('save_quality')

    # Run the simulation
    scanned_image = simulate_scan(bps_dict)
    save_image(scanned_image, output_image_path,args.dpi,  args.color_mode, args.assumed_dpi_if_missing)
    #scanned_image.convert("RGB").save(output_image_path, quality=bps_dict["save_quality"])

    # Persist the actual parameters used
    with open(output_json_path, 'w') as f:
        json.dump(bps_dict, f, indent=4)

    print(f"Wrote image: {output_image_path}")
    print(f"Wrote params: {output_json_path}")

