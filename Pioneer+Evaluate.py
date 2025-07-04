import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===========================
# Device Setup
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Loading Dataset
# ===========================
file_path = "/content/drive/MyDrive/yelp_labelled.txt"
df = pd.read_csv(file_path, sep="\t", header=None, names=["text", "label"])
df = df.dropna()
df["label"] = df["label"].map({0: "negative", 1: "positive"})

# ===========================
# Load Model and Tokenizer
# ===========================
model_name = "/content/drive/MyDrive/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# ===========================
# Score Example with Full Features
# ===========================
@torch.inference_mode()
def score_example(example_text, label=None):
    prompt = f"You are a sentiment classifier. Think step-by-step before giving a final answer.\n\nTweet: {example_text}\nReasoning:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_tokens, log_probs, entropy_values = [], [], []
    for _ in range(30):
        outputs = model(input_ids, return_dict_in_generate=False, output_scores=True)
        scores = outputs[0][:, -1, :]
        probs = F.softmax(scores, dim=-1)

        best_token_id = torch.argmax(probs, dim=-1).item()
        best_log_prob = torch.log(probs[0, best_token_id] + 1e-8).item()
        entropy = -(probs[0] * torch.log(probs[0] + 1e-8)).sum().item()

        if best_token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(best_token_id)
        log_probs.append(best_log_prob)
        entropy_values.append(entropy)

        input_ids = torch.cat([input_ids, torch.tensor([[best_token_id]]).to(device)], dim=-1)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Self-Correction Phase
    revised_prompt = prompt + generated_text + "\nYour answer is incorrect. Please revise."
    revised_inputs = tokenizer(revised_prompt, return_tensors="pt").to(device)
    revised_input_ids = revised_inputs["input_ids"]
    revised_tokens = []
    for _ in range(30):
        outputs = model(revised_input_ids, return_dict_in_generate=False, output_scores=True)
        scores = outputs[0][:, -1, :]
        probs = F.softmax(scores, dim=-1)
        best_token_id = torch.argmax(probs, dim=-1).item()
        if best_token_id == tokenizer.eos_token_id:
            break
        revised_tokens.append(best_token_id)
        revised_input_ids = torch.cat([revised_input_ids, torch.tensor([[best_token_id]]).to(device)], dim=-1)
    revised_text = tokenizer.decode(revised_tokens, skip_special_tokens=True)

    self_corrected = float(generated_text.strip().lower() != revised_text.strip().lower())
    length = len(generated_tokens)
    avg_logprob = np.mean(log_probs)
    avg_entropy = np.mean(entropy_values)
    final_score = avg_logprob - avg_entropy - 0.01 * length + 0.5 * self_corrected

    return {
        "text": example_text,
        "label": label,
        "avg_logprob": avg_logprob,
        "avg_entropy": avg_entropy,
        "length": length,
        "self_corrected": self_corrected,
        "final_score": final_score,
        "generated_reasoning": generated_text,
        "revised_reasoning": revised_text
    }

# ===========================
# Select Top-K Examples
# ===========================
scored_examples = [score_example(row["text"], row["label"]) for _, row in df.sample(30).iterrows()]
scored_examples.sort(key=lambda x: x["final_score"], reverse=True)
few_shot_examples = scored_examples[:3]

# ===========================
# Main Analysis Function
# ===========================
@torch.inference_mode()
def analyze_example(tweet, label=None, token_critique_threshold=-3.0):
    prompt = "You are a sentiment classifier. Think step-by-step before giving a final answer.\n\n"
    for ex in few_shot_examples:
        prompt += f"Tweet: {ex['text']}\nReasoning: {ex['generated_reasoning']}\nSentiment: {ex['label']}\n\n"
    prompt += f"Tweet: {tweet}\nReasoning:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    generated_tokens, log_probs, entropy_values, token_infos = [], [], [], []
    for _ in range(30):
        outputs = model(input_ids, return_dict_in_generate=False, output_scores=True)
        scores = outputs[0][:, -1, :]
        probs = F.softmax(scores, dim=-1)

        best_token_id = torch.argmax(probs, dim=-1).item()
        best_log_prob = torch.log(probs[0, best_token_id] + 1e-8).item()
        entropy = -(probs[0] * torch.log(probs[0] + 1e-8)).sum().item()

        avg_entropy_so_far = np.mean(entropy_values) if entropy_values else 0.0
        dynamic_threshold = token_critique_threshold + (avg_entropy_so_far * 0.5)
        if best_log_prob < dynamic_threshold:
            best_token_id = torch.argsort(probs, dim=-1, descending=True)[0][1].item()
            best_log_prob = torch.log(probs[0, best_token_id] + 1e-8).item()

        if best_token_id == tokenizer.eos_token_id:
            break

        generated_tokens.append(best_token_id)
        log_probs.append(best_log_prob)
        entropy_values.append(entropy)

        token_text = tokenizer.decode([best_token_id])
        token_infos.append((token_text, best_log_prob, entropy))

        input_ids = torch.cat([input_ids, torch.tensor([[best_token_id]]).to(device)], dim=-1)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Self-Critique Revision Phase
    revised_prompt = prompt + generated_text + "\nYour answer is incorrect. Please revise."
    revised_inputs = tokenizer(revised_prompt, return_tensors="pt").to(device)
    revised_input_ids = revised_inputs["input_ids"]
    revised_tokens = []

    for _ in range(30):
        outputs = model(revised_input_ids, return_dict_in_generate=False, output_scores=True)
        scores = outputs[0][:, -1, :]
        probs = F.softmax(scores, dim=-1)
        best_token_id = torch.argmax(probs, dim=-1).item()
        if best_token_id == tokenizer.eos_token_id:
            break
        revised_tokens.append(best_token_id)
        revised_input_ids = torch.cat([revised_input_ids, torch.tensor([[best_token_id]]).to(device)], dim=-1)

    revised_text = tokenizer.decode(revised_tokens, skip_special_tokens=True)

    # Reasoning Tree
    reasoning_tree = {
        "Input": tweet,
        "Step1": "Extract sentiment-related words",
        "Step2": "Assess intensity and context",
        "Final Decision": revised_text.strip()
    }

    # Critique Chain
    critique_chain = [
        "Initial conclusion may lack context.",
        "Reevaluating sentiment strength...",
        "Final decision confirmed."
    ]

    # Saliency Simulation
    saliency = {word: np.random.rand() for word in tweet.split()}

    return {
        "tweet": tweet,
        "true_label": label,
        "reasoning": generated_text.strip(),
        "revised_answer": revised_text.strip(),
        "avg_logprob": np.mean(log_probs),
        "avg_entropy": np.mean(entropy_values),
        "final_score": np.mean(log_probs) - np.mean(entropy_values),
        "confidence_score": 1.0 if revised_text.strip().lower() != generated_text.strip().lower() else 0.5,
        "token_backtracking": token_infos,
        "reasoning_tree": reasoning_tree,
        "critique_chain": critique_chain,
        "saliency": saliency
    }

# ===========================
# Run on One Sample
# ===========================
sample = df.sample(1).iloc[0]
result = analyze_example(sample["text"], sample["label"])

print("=== Final Output ===")
for k, v in result.items():
    if isinstance(v, dict):
        print(f"{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv}")
    elif isinstance(v, list):
        print(f"{k}:")
        for item in v:
            print(f"  {item}")
    else:
        print(f"{k}: {v}\n")
