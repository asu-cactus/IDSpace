import os
import json
import argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import torch

def generate_text_from_image(image_path, model, processor):
    try:
        image = Image.open(image_path)
        inputs = processor.process(
            images=[image],
            text="Can you describe the attributes of this image? The attributes may include: the type of document (e.g., passport, driver's license, identity card, etc.); gender of person in document (e.g., male or female); country or state of the document (e.g., California, Serbia, etc.)"
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

def main(args):
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
    for root, dirs, files in os.walk(args.image_directory):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Processing {image_path}...")
                generated_text = generate_text_from_image(image_path, model, processor)
                if generated_text:
                    results[image_path] = generated_text

    with open(args.output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document attribute extraction with Molmo-7B.")
    parser.add_argument('--image_directory', type=str, required=True, help='Path to the root image directory.')
    parser.add_argument('--output_json_path', type=str, required=True, help='Path to save the output JSON.')
    args = parser.parse_args()
    main(args)
