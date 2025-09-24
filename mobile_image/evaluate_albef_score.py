import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, ALBEFForImageTextRetrieval

def compute_albef_scores(json_path, image_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model_name = "Salesforce/albef-base"
    processor = AutoProcessor.from_pretrained(model_name)
    model = ALBEFForImageTextRetrieval.from_pretrained(model_name).to(device)
    model.eval()

    with open(json_path, 'r') as f:
        data = json.load(f)

    total_score = 0.0
    count = 0

    for rel_path, text in tqdm(data.items(), desc="Evaluating ALBEF matching scores"):
        image_path = rel_path if os.path.isabs(rel_path) else os.path.join(image_dir, rel_path)
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                score = outputs.logits_per_image.item()  # Matching score
                total_score += score
                count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if count == 0:
        print("No valid image-text pairs found.")
    else:
        avg_score = total_score / count
        print(f"\nAverage ALBEF matching score across {count} image-text pairs: {avg_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALBEF matching scores between image-text pairs")
    parser.add_argument("--json_path", required=True, help="Path to JSON file with image-text pairs")
    parser.add_argument("--image_dir", required=True, help="Base directory for resolving image paths")
    args = parser.parse_args()

    compute_albef_scores(args.json_path, args.image_dir)
