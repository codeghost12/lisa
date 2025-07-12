import torch
import numpy as np
import random
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Load model/tokenizer only once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)
model.eval()

# === Compute log-likelihood
def compute_log_likelihood(text: str) -> float:
    if not text.strip():
        return float("-inf")
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        return -outputs.loss.item() * input_ids.size(1)

# === Perturbation
def perturb_text(text: str, num_perturbations: int = 10) -> list:
    words = text.split()
    perturbed = []
    for _ in range(num_perturbations):
        w = words.copy()
        if len(w) > 5:
            del w[random.randint(0, len(w)-1)]
        perturbed.append(" ".join(w))
    return perturbed

# === DetectGPT score
def detectgpt_score(text: str, num_perturbations: int = 10) -> float:
    base_ll = compute_log_likelihood(text)
    perturbed_lls = [compute_log_likelihood(p) for p in perturb_text(text, num_perturbations)]
    return base_ll - np.mean(perturbed_lls)

# === Evaluation on human:AI ratios
def evaluate_detectgpt_on_ratios(
    texts, labels, threshold=0.0,
    ratios=[(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
):
    texts = np.array(texts)
    labels = np.array(labels)
    results = []

    for human_ratio, ai_ratio in ratios:
        print(f"\n=== Evaluating with human:ai = {human_ratio}:{ai_ratio} ===")

        human_idx = np.where(labels == 0)[0]
        ai_idx = np.where(labels == 1)[0]

        total_samples = int(min(len(human_idx) / human_ratio, len(ai_idx) / ai_ratio))
        num_human = int(total_samples * human_ratio)
        num_ai = int(total_samples * ai_ratio)

        selected_human = np.random.choice(human_idx, num_human, replace=False)
        selected_ai = np.random.choice(ai_idx, num_ai, replace=False)
        selected_idx = np.concatenate([selected_human, selected_ai])

        selected_texts = texts[selected_idx]
        selected_labels = labels[selected_idx]

        # Stratified split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            selected_texts, selected_labels, test_size=0.2, stratify=selected_labels, random_state=42
        )

        # Print class distributions
        print(f"Training samples: {len(train_texts)}")
        print(f"  -> Human: {np.sum(train_labels == 0)}")
        print(f"  -> AI: {np.sum(train_labels == 1)}")
        print(f"Evaluation samples: {len(test_texts)}")
        print(f"  -> Human: {np.sum(test_labels == 0)}")
        print(f"  -> AI: {np.sum(test_labels == 1)}")

        preds = []
        scores = []

        for i, text in enumerate(test_texts):
            print(f"Scoring {i+1}/{len(test_texts)}")
            score = detectgpt_score(text)
            pred = 1 if score < threshold else 0
            preds.append(pred)
            scores.append(score)

        print("\nClassification Report:")
        print(classification_report(test_labels, preds, target_names=["Human", "AI"]))

        result = {
            "human_ratio": human_ratio,
            "ai_ratio": ai_ratio,
            "accuracy": np.mean(np.array(preds) == test_labels),
            "precision_AI": precision_score(test_labels, preds, pos_label=1),
            "recall_AI": recall_score(test_labels, preds, pos_label=1),
            "f1_AI": f1_score(test_labels, preds, pos_label=1),
            "precision_human": precision_score(test_labels, preds, pos_label=0),
            "recall_human": recall_score(test_labels, preds, pos_label=0),
            "f1_human": f1_score(test_labels, preds, pos_label=0),
        }
        results.append(result)

    return pd.DataFrame(results)


def evaluate_detectgpt(texts, labels, threshold=0.0):
    texts = np.array(texts)
    labels = np.array(labels)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    preds = []
    scores = []

    for i, text in enumerate(test_texts):
        print(f"Scoring {i+1}/{len(test_texts)}")
        score = detectgpt_score(text)
        pred = 1 if score < threshold else 0
        preds.append(pred)
        scores.append(score)

    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=["Human", "AI"]))

    return {
        "accuracy": np.mean(np.array(preds) == test_labels),
        "precision_AI": precision_score(test_labels, preds, pos_label=1),
        "recall_AI": recall_score(test_labels, preds, pos_label=1),
        "f1_AI": f1_score(test_labels, preds, pos_label=1),
        "precision_human": precision_score(test_labels, preds, pos_label=0),
        "recall_human": recall_score(test_labels, preds, pos_label=0),
        "f1_human": f1_score(test_labels, preds, pos_label=0),
    }
