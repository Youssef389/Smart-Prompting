from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np


model_name = "/content/drive/MyDrive/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

few_shot_examples = [
    {
        "tweet": "I love this airline, they always take care of me!",
        "label": "positive",
    },
    {
        "tweet": "@JetBlue ruined my vacation. Worst airline ever.",
        "label": "negative",
    },
    {
        "tweet": "The flight was okay, nothing special.",
        "label": "neutral",
    }
]

new_example = {
    "tweet": "@JetBlue You’ve been amazing through this whole thing!"
}


def analyze_example(tweet, label=None):
    prompt = f"You are a sentiment classifier. Think step-by-step before giving a final answer.\n\n"

    for ex in few_shot_examples:
        prompt += f"Tweet: {ex['tweet']}\n"
        prompt += f"Reasoning: The tweet expresses {ex['label']} sentiment.\n"
        prompt += f"Sentiment: {ex['label']}\n\n"

    prompt += f"Tweet: {tweet}\nReasoning:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=50,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )

    generated_ids = output.sequences[0]
    generated_text = tokenizer.decode(generated_ids[len(input_ids[0]):], skip_special_tokens=True)

    scores = output.scores
    probs = [F.softmax(score[0], dim=-1) for score in scores]
    log_probs = []
    entropy = []

    for i, p in enumerate(probs):
        token_id = generated_ids[len(input_ids[0]) + i]
        log_probs.append(torch.log(p[token_id]).item())
        entropy.append(-(p * torch.log(p)).sum().item())

    return {
        "reasoning": generated_text.strip(),
        "log_probs": log_probs,
        "entropy": entropy,
        "avg_logprob": np.mean(log_probs),
        "avg_entropy": np.mean(entropy),
        "weak_steps": sum(lp < -11.0 and ent > 6.0 for lp, ent in zip(log_probs, entropy)),
    }

all_examples = few_shot_examples + [new_example]
results = []

for ex in all_examples:
    res = analyze_example(ex["tweet"], ex.get("label"))
    res["tweet"] = ex["tweet"]
    res["label"] = ex.get("label", "N/A")
    res["final_score"] = res["avg_logprob"] - res["avg_entropy"]
    results.append(res)

sorted_examples = sorted(results[:-1], key=lambda x: x["final_score"], reverse=True)[:3]

print("=== Selected Examples ===\n")
for ex in sorted_examples:
    print(f"Tweet: {ex['tweet']}")
    print(f"Label: {ex['label']}")
    print(f"Reasoning: {ex['reasoning'].splitlines()[0]}")
    print(f"Score: {ex['final_score']:.4f}")
    print("-" * 50)

print("\n=== Evaluation ===")
print(f"Tweet: {new_example['tweet']}")
print(f"Reasoning: {results[-1]['reasoning']}")
print(f"Predicted Sentiment: {results[-1]['reasoning'].split('Sentiment:')[-1].strip()}")
print(f"Final Score: {results[-1]['final_score']:.4f}")
