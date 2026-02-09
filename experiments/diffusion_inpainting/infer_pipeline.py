import os, json, torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
#from diffusers.models import UNet2DConditionModel
from safetensors.torch import load_file as safetensors_load
import sys

CROP = (650, 0, 650 + 512, 512)
number = sys.argv[1]
num_images_per_prompt = 8

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
UNET_CKPT_DIR = f"outputs/lora_{number}/epoch_4"
TEST_JSONL = "outputs/inpaint_ALB_surname_test/test.jsonl"
OUT_DIR = f"outputs/infer_lora_{number}"

NUM_STEPS = 100
GUIDANCE = 8
STRENGTH = 0.8
SEED = 12345

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "crop"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "full"), exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load base pipeline (this creates a UNet with real parameter tensors — not meta)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    safety_checker=None,
)
# keep on CPU for now
pipe.to("cpu")

# 2) Find your checkpoint file inside UNET_CKPT_DIR
safetensors_path = None
bin_path = None
for fn in os.listdir(UNET_CKPT_DIR):
    if fn.endswith(".safetensors"):
        safetensors_path = os.path.join(UNET_CKPT_DIR, fn)
    if fn.endswith(".bin") or fn.endswith(".pt") or fn.endswith(".pth"):
        bin_path = os.path.join(UNET_CKPT_DIR, fn)

if safetensors_path is None and bin_path is None:
    raise FileNotFoundError(f"No safetensors/.bin found in {UNET_CKPT_DIR}")

# 3) Load checkpoint into a CPU state dict, then load into pipe.unet
print("Loading checkpoint into CPU memory...")
if safetensors_path is not None:
    # safetensors -> returns dict of tensors (already on CPU)
    sd = safetensors_load(safetensors_path, device="cpu")
else:
    # fallback to torch.load (may be large)
    print("No safetensors found; loading torch .bin/.pt file (may be slower).")
    sd = torch.load(bin_path, map_location="cpu")

# sd keys might be prefixed (e.g., "unet.") depending on how you saved.
# We will try a few heuristics to match the pipe.unet state_dict keys.

unet_state = pipe.unet.state_dict()

def try_load_state_dict(target_module, sd_dict):
    """
    Try to load sd_dict into target_module with a few common key-shape fixes.
    Returns True if load_state_dict succeeded (strict=False) and many keys matched.
    """
    try:
        missing, unexpected = target_module.load_state_dict(sd_dict, strict=False)
        # load_state_dict returns a NamedTuple in new PyTorch; check missing/unexpected
        # We'll be conservative: consider it successful if not ALL keys are missing.
        matched = len(sd_dict) - len(unexpected)
        total = len(unet_state)
        print(f"Loaded state_dict with {matched}/{total} matched keys (len(sd)={len(sd_dict)})")
        return True
    except Exception as e:
        print("load_state_dict failed:", e)
        return False

# Heuristic 1: keys match exactly
if try_load_state_dict(pipe.unet, sd):
    print("Loaded checkpoint directly into pipe.unet.")
else:
    # Heuristic 2: maybe keys are saved with prefix like "unet."
    prefixed = {}
    for k, v in sd.items():
        newk = k
        if not k.startswith("unet."):
            newk = "unet." + k
        prefixed[newk] = v
    if try_load_state_dict(pipe.unet, prefixed):
        print("Loaded checkpoint after adding 'unet.' prefix.")
    else:
        # Heuristic 3: maybe your checkpoint is already a full model state (no prefix)
        # or contains "model." prefix etc. Try stripping common prefixes present in sd keys.
        any_key = next(iter(sd.keys()))
        print("Example checkpoint key:", any_key)
        # Try to strip up to first dot
        stripped = {}
        for k, v in sd.items():
            if "." in k:
                stripped_k = ".".join(k.split(".")[1:])
            else:
                stripped_k = k
            stripped[stripped_k] = v
        if try_load_state_dict(pipe.unet, stripped):
            print("Loaded checkpoint after stripping first prefix segment.")
        else:
            # Give up with informative error
            raise RuntimeError(
                "Could not match checkpoint keys to pipe.unet. "
                "Inspect the checkpoint key names and pipe.unet.state_dict().keys()."
            )

# 4) Move pipeline to GPU (or desired device) AFTER loading real tensors
pipe.to(device)
pipe.unet.eval()

# Optional: fuse LoRA if present (not needed here since you loaded full unet)
# pipe.fuse_lora()

g = torch.Generator(device=device).manual_seed(SEED)




NEGATIVE_PROMPT = (
        "background change, texture change, color shift, lighting change, "
        "blur, noise, artifacts, distorted text, extra characters, "
        "wrong font, misalignment, watermark"
    
)

POSITIVE_PROMPT_PATCH = ". Keep the same font, size, alignment, and ink color. Do not change any other text or background."




with open(TEST_JSONL, "r", encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]
for i, row in enumerate(rows[:1]):
    full_src = Image.open(row["source"]).convert("RGB")
    full_msk = Image.open(row["mask"]).convert("L")
    prompt = row["prompt"]
    #prompt += POSITIVE_PROMPT_PATCH
    print(prompt)

    src = full_src.crop(CROP)
    msk = full_msk.crop(CROP)

    out_crop = pipe(
        prompt=prompt,
        #negative_prompt=NEGATIVE_PROMPT,
        image=src,
        mask_image=msk,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        strength=STRENGTH,
        generator=g,
        num_images_per_prompt= num_images_per_prompt,
    ).images

    basename = os.path.basename(row['source'])
    base = os.path.splitext(os.path.basename(row["source"]))[0]
    crop_path = os.path.join(OUT_DIR, "crop", f"generated_{basename}")
    for ii in range(num_images_per_prompt):
    #out_crop[i].save(crop_path)

    # paste back into full-size image
        full_out = full_src.copy()
        full_out.paste(out_crop[ii], (CROP[0], CROP[1]))
        full_path = os.path.join(OUT_DIR, "full", f"{str(ii)}_generated_{basename}")
        full_out.save(full_path)

    print(f"[{i+1}/{len(rows)}] saved {crop_path} and {full_path}")

print("Done.")




