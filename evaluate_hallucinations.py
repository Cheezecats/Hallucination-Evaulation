import requests
import json
from tqdm import tqdm

# ---- Configuration ----
API_URL = "http://127.0.0.1:1234/v1/chat/completions"
TEMPERATURE = 0.7
MAX_TOKENS = 300

# ---- Load Questions and Answers ----
def load_qa_pairs(filename):
    qa_pairs = []
    current_q = None
    
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Q: "):
                current_q = line[3:]  # Remove "Q: "
            elif line.startswith("A: ") and current_q is not None:
                qa_pairs.append((current_q, line[3:]))  # Remove "A: "
                current_q = None
    return qa_pairs

try:
    qa_pairs = load_qa_pairs("simpleqa_benchmark.txt")
    questions = [q for q, a in qa_pairs]
    ground_truths = [a for q, a in qa_pairs]
    
    if not qa_pairs:
        print("❌ Error: No valid Q/A pairs found in the file!")
        exit(1)
    print(f"✅ Loaded {len(qa_pairs)} Q/A pairs")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit(1)

# ---- Ask user which model to evaluate ----
print("\nAvailable models (adjust if needed): gemma-1b, gemma-4b, gemma-12b, gemma-27b")
model_name = input("Enter the model name you want to evaluate (e.g., 'gemma-7b'): ").strip()

# ---- Generate Responses ----
def generate_response(question, model_name):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": question}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response_json = response.json()
        
        # Handle different response formats
        if "choices" in response_json:
            return response_json["choices"][0]["message"]["content"]
        elif "content" in response_json:  # Alternative format
            return response_json["content"]
        else:
            print(f"⚠️ Unexpected response format: {response_json}")
            return ""
    except Exception as e:
        print(f"⚠️ Error generating response: {e}")
        return ""
# ---- Evaluate Hallucinations ----
def is_hallucination(response, truth):
    # Simple check: ground truth should appear in the response
    return truth.lower() not in response.lower()

# ---- Run Evaluation ----
print(f"\n🚀 Evaluating {model_name}...")
responses = []
for question in tqdm(questions, desc="Generating responses"):
    response = generate_response(question, model_name)
    responses.append(response)

# Calculate hallucination rate
hallucinations = [is_hallucination(resp, truth) for resp, truth in zip(responses, ground_truths)]
hallucination_rate = sum(hallucinations) / len(hallucinations)

# Save raw responses
with open(f"responses_{model_name}.json", "w") as f:
    json.dump(responses, f)

# ---- Print Results ----
print(f"\n📊 Results for {model_name}:")
print(f"- Hallucination rate: {hallucination_rate:.2%}")
print(f"- Correct answers: {len(hallucinations) - sum(hallucinations)}/{len(hallucinations)}")
print(f"- Responses saved to: responses_{model_name}.json")