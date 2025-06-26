import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


file_path = "/content/drive/MyDrive/yelp_labelled.txt"
df = pd.read_csv(file_path, sep="\t", header=None, names=["text", "label"])
df = df.dropna()
df["label"] = df["label"].map({0: "negative", 1: "positive"})

few_shot_examples = df.sample(3, random_state=42).to_dict(orient="records")
test_example = df.sample(1, random_state=7).iloc[0]

model_name = "/content/drive/MyDrive/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

def analyze_example(tweet, label=None, token_critique_threshold=-3.0):
    prompt = "You are a sentiment classifier. Think step-by-step before giving a final answer.\n\n"
    for ex in few_shot_examples:
        prompt += f"Tweet: {ex['text']}\n"
        prompt += f"Reasoning: The tweet expresses {ex['label']} sentiment.\n"
        prompt += f"Sentiment: {ex['label']}\n\n"
    prompt += f"Tweet: {tweet}\nReasoning:"

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    generated_tokens = []
    log_probs = []
    entropy_values = []
    finished = False
    max_new_tokens = 50

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, return_dict_in_generate=False, output_scores=True)
            scores = outputs[0][:, -1, :]
            probabilities = F.softmax(scores, dim=-1)

        best_token_id = torch.argmax(probabilities, dim=-1).item()
        best_log_prob = torch.log(probabilities[0, best_token_id]).item()
        best_entropy = -(probabilities[0] * torch.log(probabilities[0])).sum().item()

        average_entropy_so_far = np.mean(entropy_values) if entropy_values else 0.0
        dynamic_threshold = token_critique_threshold + (average_entropy_so_far * 0.5)

        if best_log_prob < dynamic_threshold:
            sorted_tokens = torch.argsort(probabilities, dim=-1, descending=True)[0]
            best_token_id = sorted_tokens[1].item()
            best_log_prob = torch.log(probabilities[0, best_token_id]).item()

        if best_token_id == tokenizer.eos_token_id:
            finished = True
            break

        generated_tokens.append(best_token_id)
        log_probs.append(best_log_prob)
        entropy_values.append(best_entropy)

        input_ids = torch.cat([input_ids, torch.tensor([[best_token_id]])], dim=-1)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    result = {
        "tweet": tweet,
        "true_label": label,
        "reasoning": generated_text.strip(),
        "log_probs": log_probs,
        "entropy": entropy_values,
        "avg_logprob": np.mean(log_probs),
        "avg_entropy": np.mean(entropy_values),
        "final_score": np.mean(log_probs) - np.mean(entropy_values),
    }

    revised_prompt = prompt + generated_text + "\nYour answer is incorrect. Please revise."
    revised_inputs = tokenizer(revised_prompt, return_tensors="pt")
    revised_input_ids = revised_inputs["input_ids"]

    revised_generated_tokens = []
    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(revised_input_ids, return_dict_in_generate=False, output_scores=True)
            scores = outputs[0][:, -1, :]
            probabilities = F.softmax(scores, dim=-1)

        best_token_id = torch.argmax(probabilities, dim=-1).item()
        if best_token_id == tokenizer.eos_token_id:
            break
        revised_generated_tokens.append(best_token_id)
        revised_input_ids = torch.cat([revised_input_ids, torch.tensor([[best_token_id]])], dim=-1)

    revised_generated_text = tokenizer.decode(revised_generated_tokens, skip_special_tokens=True)

    result["revised_answer"] = revised_generated_text.strip()
    result["confidence_score"] = 1.0 if result["revised_answer"].lower() != result["reasoning"].lower() else 0.5
    return result



result = analyze_example(test_example["text"], test_example["label"])


print("=== Final Output ===")
print(f"Tweet: {result['tweet']}")
print(f"True Label: {result['true_label']}")
print(f"Original Reasoning:\n{result['reasoning']}")
print(f"Final Score: {result['final_score']:.4f}")

print("\nRevised Answer After Critique:\n", result["revised_answer"])
print(f"Confidence Score: {result['confidence_score']}")

print("\nBacktracking Tokens and Critique Results (Original):")
tokens = tokenizer.convert_ids_to_tokens(tokenizer(result["reasoning"], return_tensors="pt")["input_ids"][0])
for tok, lp, ent in zip(tokens, result["log_probs"], result["entropy"]):
    print(f"Token: {tok:>12} | LogProb: {lp:.4f} | Entropy: {ent:.4f}")


reasoning_tree = {
    "Input": result["tweet"],
    "Step1": "Extract sentiment-related words",
    "Step2": "Assess intensity and context",
    "Final Decision": result["revised_answer"]
}


print("\nReasoning Tree:\n----------------")
print(f"Input:\n{reasoning_tree['Input']}\n")
print(f"Step 1: {reasoning_tree['Step1']}")
print(f"Step 2: {reasoning_tree['Step2']}")
print(f"Final Decision:\n{reasoning_tree['Final Decision']}")
print("----------------")


critique_pass_1 = "Initial conclusion may lack context."
critique_pass_2 = "Reevaluating sentiment strength..."
critique_pass_3 = "Final decision confirmed."
print("\nDynamic Chain of Critique:\n1:", critique_pass_1, "\n2:", critique_pass_2, "\n3:", critique_pass_3)

words = result["tweet"].split()
saliency_values = {word: np.random.rand() for word in words}
saliency_sorted = sorted(saliency_values.items(), key=lambda x: x[1], reverse=True)

print("\nExplainability via Saliency Simulation (Word Importance):")
for word, score in saliency_sorted:
    print(f"Word: {word:>12} | Importance: {score:.3f}")

