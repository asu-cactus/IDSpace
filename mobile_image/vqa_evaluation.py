import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration, pipeline

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vqa_model_name = "Salesforce/blip2-flan-t5-xl"  # Supports VQA
vqa_model = Blip2ForConditionalGeneration.from_pretrained(vqa_model_name).to(device)
vqa_processor = Blip2Processor.from_pretrained(vqa_model_name)

# Use a small LLM for local or mock question generation, or use OpenAI/GPT API if available
from transformers import AutoModelForCausalLM, AutoTokenizer

# You can replace this with GPT-3.5 call if desired
llm_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
llm_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct").to(device)

def generate_questions(text, num_questions=10):
    prompt = (
        f"Based on the following image description, generate {num_questions} detailed yes/no questions "
        "from different perspectives: device type, lighting conditions, background content, camera angle, etc.\n"
        f"Description:\n{text}\nQuestions:"
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = llm_model.generate(**inputs, max_new_tokens=300)
    questions_block = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    questions = [q.strip("- ").strip() for q in questions_block.split("\n") if "?" in q]
    return questions[:num_questions]

def vqa_answer(image, question):
    inputs = vqa_processor(image=image, text=question, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = vqa_model.generate(**inputs, max_new_tokens=10)
        answer = vqa_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer.lower().strip()

def evaluate_vqa_similarity(json_path, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_accuracy = 0.0
    count = 0

    for rel_path, text in tqdm(data.items(), desc="Evaluating VQA-based similarity"):
        image_path = rel_path if os.path.isabs(rel_path) else os.path.join(image_dir, rel_path)
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
            questions = generate_questions(text)

            correct = 0
            for q in questions:
                vqa_resp = vqa_answer(image, q)
                if any(affirm in vqa_resp for affirm in ["yes", "yeah", "true"]):  # Simple match
                    correct += 1

            score = correct / len(questions)
            total_accuracy += score
            count += 1
        except Exception as e:
            print(f"Error with {image_path}: {e}")

    if count == 0:
        print("No valid image-text pairs evaluated.")
    else:
        avg_accuracy = total_accuracy / count
        print(f"\nAverage VQA-based similarity accuracy across {count} image-text pairs: {avg_accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA-based similarity between image-text pairs")
    parser.add_argument("--json_path", required=True, help="Path to JSON file of image-text pairs")
    parser.add_argument("--image_dir", required=True, help="Base path for resolving image paths")
    args = parser.parse_args()

    evaluate_vqa_similarity(args.json_path, args.image_dir)
