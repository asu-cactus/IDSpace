from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from run_final import main

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_embeddings(text, model, tokenizer):
    """
    Compute normalized embeddings for a given text using the model and tokenizer.
    """
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1)


def find_closest_match(input_text, json_data, model, tokenizer):
    """
    Find the closest matching description in the JSON data to the input text.
    Return the image path, matched description, and similarity score.
    """
    input_embedding = compute_embeddings([input_text], model, tokenizer)
    highest_similarity = 0
    best_match = None
    best_image_path = None

    for image_path, description in json_data.items():
        description_embedding = compute_embeddings([description], model, tokenizer)
        similarity = torch.cosine_similarity(input_embedding, description_embedding).item()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = description
            best_image_path = image_path

    return best_image_path, best_match, highest_similarity


def load_json(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r') as file:
        return json.load(file)


def interactive_matching(json_file1, json_file2, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Interactive function to find matching descriptions for user input from JSON files.
    First input matches only `json_file1`, second input matches only `json_file2`.
    """
    # Load JSON data
    json_data1 = load_json(json_file1)
    json_data2 = load_json(json_file2)

    # Load Hugging Face model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print("Matching against the first JSON file...")
    user_input1 = input("Enter a text to find the closest match in JSON file 1 (or type 'exit' to quit): ").strip()
    if user_input1.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        return

    path1, match1, score1 = find_closest_match(user_input1, json_data1, model, tokenizer)
    if path1:
        print(f"\nClosest Match in JSON File 1: {match1}")
        print(f"Image Path: {path1}")
        print(f"Similarity Score: {score1:.2f}")
    else:
        print("No match found in the first JSON file.")

    print("\nNow matching against the second JSON file...")
    user_input2 = input("Enter a text to find the closest match in JSON file 2 (or type 'exit' to quit): ").strip()
    if user_input2.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        return

    path2, match2, score2 = find_closest_match(user_input2, json_data2, model, tokenizer)
    if path2:
        print(f"\nClosest Match in JSON File 2: {match2}")
        print(f"Image Path: {path2}")
        print(f"Similarity Score: {score2:.2f}")
    else:
        print("No match found in the second JSON file.")

    return path1, path2


# Example usage
json_file1 = './id_dataset.json'
json_file2 = './dataset.json'

path1, path2 = interactive_matching(json_file1, json_file2)
main(path2, path1)