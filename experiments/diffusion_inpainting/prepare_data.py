import os, json
from PIL import Image, ImageDraw
import tqdm

area = "ALB"
segment_key = "surname"
real_paths = "/path/to/data/templates/Images/reals"
annotation_path = "/path/to/annotations/ALB_original_annotation.json"   # <-- set this to your actual file
split_data_path = "/path/to/data/inputs/template_guided_datas.json"

out_root = f"outputs/inpaint_{area}_{segment_key}"
out_src = os.path.join(out_root, "src")
out_tgt = os.path.join(out_root, "tgt")
out_msk = os.path.join(out_root, "mask")
os.makedirs(out_src, exist_ok=True)
os.makedirs(out_tgt, exist_ok=True)
os.makedirs(out_msk, exist_ok=True)

jsonl_out = os.path.join(out_root, "train.jsonl")


def load_annotations(annotation_path):
    with open(annotation_path, "r") as f:
        annotations = json.load(f)
    return annotations


def clamp_bbox(b, W, H):
    """b = [x1,y1,x2,y2] possibly floats; clamp to image bounds and int."""
    x1, y1, x2, y2 = b
    x1 = int(max(0, min(W - 1, round(x1 - 20))))
    y1 = int(max(0, min(H - 1, round(y1 - 20))))
    x2 = int(max(0, min(W,     round(x2 + 20))))
    y2 = int(max(0, min(H,     round(y2 + 20))))
    # Ensure non-empty
    if x2 <= x1: x2 = min(W, x1 + 1)
    if y2 <= y1: y2 = min(H, y1 + 1)
    return [x1, y1, x2, y2]


def union_bbox(b1, b2):
    """Union of two bboxes [x1,y1,x2,y2]."""
    return [
        min(b1[0], b2[0]),
        min(b1[1], b2[1]),
        max(b1[2], b2[2]),
        max(b1[3], b2[3]),
    ]


def make_rect_mask(size_wh, bbox):
    """Returns L-mode mask: 255 inside bbox, 0 elsewhere."""
    W, H = size_wh
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    x1, y1, x2, y2 = bbox
    d.rectangle([x1, y1, x2, y2], fill=255)
    return mask


# ---- Load split
with open(split_data_path) as f:
    ts = json.load(f)

# ---- Build tuning/test
tuning_set = {}
for v in ts["train"]:
    if v[1] == 0:
        name = v[0].split("/")[-1]
        if name.split("_")[0].upper() == area:
            tuning_set[name] = name

for v in ts["val"]:
    if v[1] == 0:
        name = v[0].split("/")[-1]
        if name.split("_")[0].upper() == area:
            tuning_set[name] = name

test_set = {}
for v in ts["test"]:
    if v[1] == 0:
        name = v[0].split("/")[-1]
        if name.split("_")[0].upper() == area:
            test_set[name] = name

import os

def remove_keys_if_image_exists(img_dir, data_dict, exts=(".png", ".jpg", ".jpeg")):
    image_names = {
        fn: fn for fn in os.listdir(img_dir)
        if fn.lower().endswith(exts) and fn.split('_')[0].upper() == area
    }
    for k in list(image_names.keys()):
        if k in data_dict:
            del image_names[k]
    return image_names



tuning_set = remove_keys_if_image_exists(real_paths, test_set)


print("Size:", len(tuning_set), len(test_set))

# ---- Load annotations
annotations = load_annotations(annotation_path)

# ---- Choose prime from tuning_set
first_key = next(iter(tuning_set))
print(f"The first key is: {first_key}")

prime_filename = tuning_set.pop(first_key)
print(f"Prime filename removed from tuning_set: {prime_filename}")

# Prime segment info
prime_value = annotations[prime_filename][segment_key]["value"]
prime_bbox = annotations[prime_filename][segment_key]["bbox"]

# Prime image
prime_img_path = os.path.join(real_paths, prime_filename)
prime_img = Image.open(prime_img_path).convert("RGB")
W0, H0 = prime_img.size
prime_bbox = clamp_bbox(prime_bbox, W0, H0)

# ---- Iterate and create training rows
rows = []
for filename in tqdm.tqdm(tuning_set):
    if filename not in annotations:
        continue
    if segment_key not in annotations[filename]:
        continue

    values = annotations[filename]
    content = values[segment_key]["value"]
    bbox = values[segment_key]["bbox"]

    real_img_path = os.path.join(real_paths, filename)
    if not os.path.exists(real_img_path):
        continue

    real_img = Image.open(real_img_path).convert("RGB")
    W, H = real_img.size

    bbox = clamp_bbox(bbox, W, H)

    # 1) Union bbox (in real image coordinate space)
    # If your prime bbox is in a different image size, this assumes same template/layout.
    # If sizes differ, you MUST map coordinates. Here we assume same size.
    ub = union_bbox(prime_bbox, bbox)
    ub = clamp_bbox(ub, W, H)

    # 2) Build source (init) image by replacing union area of real_img with prime_img region
    # Copy the corresponding union region from prime_img; assumes same size/layout.
    src_img = real_img.copy()

    # If prime image size differs, resize prime to real size for consistent coordinates
    if prime_img.size != real_img.size:
        prime_resized = prime_img.resize((W, H), Image.BICUBIC)
    else:
        prime_resized = prime_img

    x1, y1, x2, y2 = ub
    patch = prime_resized.crop((x1, y1, x2, y2))
    src_img.paste(patch, (x1, y1))

    # 3) Mask image for union bbox
    mask_img = make_rect_mask((W, H), ub)

    # 4) Target image
    # Option A (common for supervised inpainting): target is the ORIGINAL real image (recover real surname)
    tgt_img = real_img

    # Option B (counterfactual): target is the image with PRIME surname (i.e., src_img)
    # tgt_img = src_img

    # 5) Save files
    src_path = os.path.join(out_src, filename)
    tgt_path = os.path.join(out_tgt, filename)
    msk_path = os.path.join(out_msk, filename)

    src_img.save(src_path)
    tgt_img.save(tgt_path)
    mask_img.save(msk_path)

    # 6) Prompt
    # You can pick whichever name you want the model to write in the masked region.
    # If training to recover original:
    #prompt = f"Replace the surname {prime_value} with: {content}"
    prompt = (
            f"Replace the surname {prime_value} with {content}. "
            f"Exact letters: {' '.join(list(content))}. "
            "First letter uppercase, remaining letters lowercase. "
            "Printed text, same font, size, spacing, and baseline. "
            "Do not change any other text or background."
    )

    # If training to write PRIME surname in other cards:
    # prompt = f"Replace the surname with: {prime_value}"

    rows.append({
        "source": src_path,
        "target": tgt_path,
        "mask": msk_path,
        "prompt": prompt
    })

# Write JSONL
with open(jsonl_out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Done. Wrote {len(rows)} rows to: {jsonl_out}")
print("Example row:", rows[0] if rows else "No rows generated.")

