import os
import json
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch

def generate_text_from_image(image_path, processor, model):
    try:
        image = Image.open(image_path)
        inputs = processor.process(
            images=[image],
            text=(
                "Can you describe the attributes of this image? The attributes may include: "
                "the device type used to capture the image (e.g., mobile, DSLR, laser scanner); "
                "the device model; the lighting conditions (e.g., indoor white light, outdoor sunlight); "
                "capture background (e.g., if it captured from a computer or TV screen, captured from a real ID card indoor while someone is holding it in hands, "
                "captured from a real ID card indoor while it is placed on a desk, etc.); "
                "and the camera or scan angle (e.g., mobile or DSLR camera angle, or scan angle if a laser scanner is used)."
            )
        )
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
            tokenizer=processor.tokenizer
        )
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        return processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main(image_directory, output_json_path):
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cuda'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='cuda'
    )

    results = {}
    for root, _, files in os.walk(image_directory):
        processed_subdirs = set()
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                subdir = os.path.abspath(root)
                if subdir not in processed_subdirs:
                    image_path = os.path.join(root, file)
                    print(f"Processing {image_path}...")
                    generated_text = generate_text_from_image(image_path, processor, model)
                    if generated_text:
                        results[image_path] = generated_text
                    processed_subdirs.add(subdir)

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptions for images using Molmo-7B-D-0924")
    parser.add_argument("--image_dir", required=True, help="Path to the image directory")
    parser.add_argument("--output", required=True, help="Path to save the output JSON file")
    args = parser.parse_args()
    main(args.image_dir, args.output)