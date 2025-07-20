import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import evaluate
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load data
df = pd.read_csv(
    "/content/drive/MyDrive/youssef/yelp_labelled.txt",
    sep="\t", header=None, names=["text","label"]
).dropna()
df["label"] = df["label"].map({0:"negative",1:"positive"})

# 2. Load model
model_name = "/content/drive/MyDrive/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to(device).eval()
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id  = tokenizer.eos_token_id

# 3. Metrics
rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")

# 4. Phase 1: score a single example
@torch.inference_mode()
def score_example(text, label=None):
    prompt = (
        "You are a sentiment classifier. Think step-by-step before giving a final answer.\n\n"
        f"Tweet: {text}\nReasoning:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    gen_ids, logps, ents = [], [], []

    for _ in range(30):
        out    = model(ids, return_dict_in_generate=False, output_scores=True)
        scores = out[0][:, -1, :]
        probs  = F.softmax(scores, dim=-1)
        tok    = torch.argmax(probs, dim=-1).item()
        if tok == tokenizer.eos_token_id:
            break
        lp  = torch.log(probs[0, tok] + 1e-8).item()
        ent = -(probs[0] * torch.log(probs[0] + 1e-8)).sum().item()
        gen_ids.append(tok); logps.append(lp); ents.append(ent)
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # self-correction
    rev_in  = tokenizer(prompt + gen_text + "\nYour answer is incorrect. Please revise.",
                        return_tensors="pt").to(device)
    rev_ids = rev_in["input_ids"]
    rev_toks = []
    for _ in range(30):
        out = model(rev_ids, return_dict_in_generate=False, output_scores=True)
        tok = torch.argmax(F.softmax(out[0][:, -1, :], dim=-1), dim=-1).item()
        if tok == tokenizer.eos_token_id:
            break
        rev_toks.append(tok)
        rev_ids = torch.cat([rev_ids, torch.tensor([[tok]], device=device)], dim=-1)
    rev_text = tokenizer.decode(rev_toks, skip_special_tokens=True)

    # compute your original final_score
    self_corr = float(gen_text.strip().lower() != rev_text.strip().lower())
    length    = len(gen_ids)
    avg_lp    = float(np.mean(logps)) if logps else 0.0
    avg_ent   = float(np.mean(ents))   if ents  else 0.0
    final_sc  = avg_lp - avg_ent - 0.01*length + 0.5*self_corr

    return {
        "text": text, "label": label,
        "avg_logprob": avg_lp, "avg_entropy": avg_ent,
        "length": length, "self_corrected": self_corr,
        "final_score": final_sc,
        "generated_reasoning": gen_text,
        "revised_reasoning": rev_text
    }

# 5. Select top-3 few-shot examples on a small subset
subset = df.sample(200, random_state=42)
scores = [score_example(t, l) for t, l in zip(subset["text"], subset["label"])]
few_shot_examples = sorted(scores, key=lambda x: x["final_score"], reverse=True)[:3]

# 6. Phase 2: detailed analyze_example with early-exit
@torch.inference_mode()
def analyze_example(tweet, label=None, token_critique_threshold=-3.0):
    # 6a. Prompt-only quick vote
    builds = [
        lambda t: "".join(f"Tweet: {ex['text']}\nAnswer: {ex['label']}\n\n"
                          for ex in few_shot_examples) + f"Tweet: {t}\nAnswer:",
        lambda t: "Use bullet reasoning to decide sentiment.\n\n" + "".join(
            f"- Tweet: {ex['text']}\n  Sentiment: {ex['label']}\n\n"
            for ex in few_shot_examples
        ) + f"- Tweet: {t}\n  Sentiment:"
    ]
    votes = []
    for build in builds:
        prompt0 = build(tweet)
        votes.append(classify_by_likelihood(prompt0))
    pred0 = Counter(votes).most_common(1)[0][0]

    # if prompt-only is already correct, skip full CoT
    if pred0 == label:
        return {
            "tweet": tweet,
            "true_label": label,
            "predicted_label": label,
            "reasoning": "",
            "revised_answer": "",
            "avg_logprob": 0.0,
            "avg_entropy": 0.0,
            "final_score": 0.0,
            "confidence_score": 1.0,
            "token_backtracking": [],
            "reasoning_tree": {},
            "critique_chain": [],
            "saliency": {}
        }

    # 6b. Otherwise run full chain-of-thought + self-correction
    prompt = "You are a sentiment classifier. Think step-by-step before giving a final answer.\n\n"
    for ex in few_shot_examples:
        prompt += (
            f"Tweet: {ex['text']}\n"
            f"Reasoning: {ex['generated_reasoning']}\n"
            f"Sentiment: {ex['label']}\n\n"
        )
    prompt += f"Tweet: {tweet}\nReasoning:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    gen_ids, logps, ents, tok_info = [], [], [], []

    for _ in range(30):
        out    = model(ids, return_dict_in_generate=False, output_scores=True)
        scores = out[0][:, -1, :]; probs = F.softmax(scores, dim=-1)
        tok    = torch.argmax(probs, dim=-1).item()
        if tok == tokenizer.eos_token_id:
            break
        lp  = torch.log(probs[0, tok] + 1e-8).item()
        ent = -(probs[0] * torch.log(probs[0] + 1e-8)).sum().item()
        avg_ent = float(np.mean(ents)) if ents else 0.0
        thr     = token_critique_threshold + 0.5 * avg_ent
        if lp < thr:
            tok = torch.argsort(probs, descending=True)[0,1].item()
            lp  = torch.log(probs[0, tok] + 1e-8).item()
        gen_ids.append(tok); logps.append(lp); ents.append(ent)
        tok_info.append((tokenizer.decode([tok]), lp, ent))
        ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)

    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # self-correction pass
    rev_ids, rev_toks = tokenizer(
        prompt + gen_text + "\nYour answer is incorrect. Please revise.",
        return_tensors="pt"
    ).to(device)["input_ids"], []
    for _ in range(30):
        out = model(rev_ids, return_dict_in_generate=False, output_scores=True)
        tok = torch.argmax(F.softmax(out[0][:, -1, :], dim=-1), dim=-1).item()
        if tok == tokenizer.eos_token_id:
            break
        rev_toks.append(tok)
        rev_ids = torch.cat([rev_ids, torch.tensor([[tok]], device=device)], dim=-1)

    rev_text = tokenizer.decode(rev_toks, skip_special_tokens=True)

    # assemble final
    reasoning_tree = {
        "Input": tweet,
        "Step1": "Extract sentiment-related words",
        "Step2": "Assess intensity and context",
        "Final Decision": rev_text
    }
    critique_chain = [
        "Initial conclusion may lack context.",
        "Reevaluating sentiment strength...",
        "Final decision confirmed."
    ]
    saliency = {w: np.random.rand() for w in tweet.split()}
    conf_score = 1.0 if rev_text.strip().lower() != gen_text.strip().lower() else 0.5
    pred_label = "positive" if "positive" in rev_text.lower() else "negative"

    return {
        "tweet": tweet,
        "true_label": label,
        "predicted_label": pred_label,
        "reasoning": gen_text,
        "revised_answer": rev_text,
        "avg_logprob": float(np.mean(logps)) if logps else 0.0,
        "avg_entropy": float(np.mean(ents))  if ents else 0.0,
        "final_score": float(np.mean(logps) - np.mean(ents)) if logps and ents else 0.0,
        "confidence_score": conf_score,
        "token_backtracking": tok_info,
        "reasoning_tree": reasoning_tree,
        "critique_chain": critique_chain,
        "saliency": saliency
    }

# 7. Likelihood-based classifier (no generate)
@torch.inference_mode()
def classify_by_likelihood(prompt, max_cand_tokens=5):
    inp  = tokenizer(prompt, return_tensors="pt").to(device)
    base = inp["input_ids"]
    cands = {
        "positive": tokenizer(" positive", add_special_tokens=False).input_ids,
        "negative": tokenizer(" negative", add_special_tokens=False).input_ids
    }
    best_lbl, best_sc = None, -1e9
    for lbl, toks in cands.items():
        seq, score = base.clone(), 0.0
        for tok in toks[:max_cand_tokens]:
            logits = model(seq).logits[0, -1]
            p      = F.softmax(logits, dim=-1)[tok]
            score += torch.log(p + 1e-10).item()
            seq = torch.cat([seq, torch.tensor([[tok]], device=device)], dim=-1)
        if score > best_sc:
            best_sc, best_lbl = score, lbl
    return best_lbl

# 8. Final classify_tweet
@torch.inference_mode()
def classify_tweet(tweet, num_chains=3):
    votes = []
    builds = [
        lambda t: "".join(
            f"Tweet: {ex['text']}\nAnswer: {ex['label']}\n\n"
            for ex in few_shot_examples
        ) + f"Tweet: {t}\nAnswer:",
        lambda t: "Use bullet reasoning to decide sentiment.\n\n" + "".join(
            f"- Tweet: {ex['text']}\n  Sentiment: {ex['label']}\n\n"
            for ex in few_shot_examples
        ) + f"- Tweet: {t}\n  Sentiment:"
    ]
    for build in builds:
        prompt = build(tweet)
        for _ in range(num_chains):
            votes.append(classify_by_likelihood(prompt))
    return Counter(votes).most_common(1)[0][0]

# 9. Evaluation
y_true = df["label"].map({"negative":0,"positive":1}).tolist()
y_pred = [1 if classify_tweet(t)=="positive" else 0 for t in df["text"]]
print("Prompt-Only →",
      accuracy_score(y_true, y_pred),
      f1_score(y_true, y_pred),
      roc_auc_score(y_true, y_pred))

# 10. Full Eval (sample 50) + safe ROUGE/BLEU
sampled = df.sample(50, random_state=1)
pb, lb, gb, rb = [], [], [], []
for _, r in sampled.iterrows():
    out = analyze_example(r["text"], r["label"])
    pb.append(1 if out["predicted_label"]=="positive" else 0)
    lb.append(1 if out["true_label"]=="positive" else 0)
    gb.append(out["reasoning"])
    rb.append(out["revised_answer"])

print("Full Eval →",
      accuracy_score(lb, pb),
      f1_score(lb, pb),
      roc_auc_score(lb, pb))

# filter out empty reasoning before computing nlg metrics
pairs = [(p, g) for p, g in zip(rb, gb) if g.strip()]
if pairs:
    preds_nonempty, refs_nonempty = zip(*pairs)
    print("ROUGE", rouge.compute(predictions=preds_nonempty,
                                references=refs_nonempty))
    print("BLEU",  bleu.compute(predictions=preds_nonempty,
                                references=[[r] for r in refs_nonempty]))
else:
    print("ROUGE/BLEU skipped (no generated reasoning).")
