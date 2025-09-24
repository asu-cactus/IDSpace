import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration

def compute_blip_scores(json_path, image_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Load model and processor
    model_name = "Salesforce/blip2-opt-6.7b"
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    # Load image-text pairs
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_score = 0.0
    count = 0

    for rel_path, text in tqdm(data.items(), desc="Evaluating BLIP-2 scores"):
        image_path = rel_path if os.path.isabs(rel_path) else os.path.join(image_dir, rel_path)
        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                # Encode image
                vision_outputs = model.vision_model(pixel_values=image_inputs["pixel_values"])
                image_features = vision_outputs.last_hidden_state[:, 0, :]  # CLS token
                image_embeds = model.visual_projection(image_features)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # Tokenize and encode text
            text_inputs = processor(text=[text], return_tensors="pt")
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            with torch.no_grad():
                text_outputs = model.text_decoder(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs.get("attention_mask"),
                    output_hidden_states=True,
                )
                text_hidden_state = text_outputs.hidden_states[-1][:, 0, :]  # First token of last layer
                text_embeds = text_hidden_state / text_hidden_state.norm(dim=-1, keepdim=True)

            # Cosine similarity
            score = torch.sum(image_embeds * text_embeds, dim=-1).item()
            total_score += score
            count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if count == 0:
        print("No valid image-text pairs found.")
    else:
        avg_score = total_score / count
        print(f"\nAverage BLIP-2 score across {count} image-text pairs: {avg_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BLIP-2 similarity between image-text pairs")
    parser.add_argument("--json_path", required=True, help="Path to JSON file with image-text pairs")
    parser.add_argument("--image_dir", required=True, help="Base directory for resolving image paths")
    args = parser.parse_args()

    compute_blip_scores(args.json_path, args.image_dir)
