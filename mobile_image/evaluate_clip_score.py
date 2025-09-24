import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import open_clip

def compute_clip_scores_openclip(json_path, image_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Load OpenCLIP model, preprocess, and tokenizer
    model, preprocess, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2B-s32B-b79K")
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()

    total_score = 0.0
    count = 0

    for rel_path, text in tqdm(data.items(), desc="Evaluating OpenCLIP scores"):
        image_path = rel_path if os.path.isabs(rel_path) else os.path.join(image_dir, rel_path)
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)

            # Tokenize text
            text_tokens = tokenizer([text]).to(device)

            with torch.no_grad():
                # Encode features
                image_features = model.encode_image(image_tensor)
                text_features = model.encode_text(text_tokens)

                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features).item()
                total_score += similarity
                count += 1
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if count == 0:
        print("No valid image-text pairs found.")
    else:
        avg_score = total_score / count
        print(f"\nAverage OpenCLIP score across {count} image-text pairs: {avg_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OpenCLIP similarity between image-text pairs")
    parser.add_argument("--json_path", required=True, help="Path to JSON file with image-text pairs")
    parser.add_argument("--image_dir", required=True, help="Base directory for resolving image paths")
    args = parser.parse_args()

    compute_clip_scores_openclip(args.json_path, args.image_dir)
