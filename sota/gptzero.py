import os
import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

# Load model and tokenizer once
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Calculate perplexity
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return torch.exp(loss).item()

# Classify based on perplexity threshold
def classify_text(text, threshold=40.0):
    ppl = calculate_perplexity(text)
    pred = 1 if ppl < threshold else 0  # 1 = AI, 0 = Human
    return pred, ppl

def evaluate_gptzero_on_ratios(
    texts, labels, threshold=40.0,
    ratios=[(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
):
    texts = np.array(texts)
    labels = np.array(labels)
    results = []

    for human_ratio, ai_ratio in ratios:
        print(f"\n=== Evaluating with human:ai = {human_ratio}:{ai_ratio} ===")

        human_indices = np.where(labels == 0)[0]
        ai_indices = np.where(labels == 1)[0]

        total_samples = int(min(len(human_indices) / human_ratio, len(ai_indices) / ai_ratio))
        num_human = int(total_samples * human_ratio)
        num_ai = int(total_samples * ai_ratio)

        selected_human = np.random.choice(human_indices, num_human, replace=False)
        selected_ai = np.random.choice(ai_indices, num_ai, replace=False)
        selected_indices = np.concatenate([selected_human, selected_ai])

        selected_texts = texts[selected_indices]
        selected_labels = labels[selected_indices]

        # Train/test split
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            selected_texts, selected_labels, test_size=0.2, stratify=selected_labels, random_state=42
        )

        # Print class counts
        print(f"Training samples: {len(train_texts)}")
        print(f"  -> Human: {np.sum(train_labels == 0)}")
        print(f"  -> AI: {np.sum(train_labels == 1)}")
        print(f"Evaluation samples: {len(eval_texts)}")
        print(f"  -> Human: {np.sum(eval_labels == 0)}")
        print(f"  -> AI: {np.sum(eval_labels == 1)}")

        y_true, y_pred, perplexities = [], [], []

        for i, text in enumerate(eval_texts):
            print(f"Scoring {i+1}/{len(eval_texts)}")
            pred, ppl = classify_text(text, threshold=threshold)
            y_true.append(eval_labels[i])
            y_pred.append(pred)
            perplexities.append(ppl)

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

        result = {
            "human_ratio": human_ratio,
            "ai_ratio": ai_ratio,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_AI": precision_score(y_true, y_pred, pos_label=1),
            "recall_AI": recall_score(y_true, y_pred, pos_label=1),
            "f1_AI": f1_score(y_true, y_pred, pos_label=1),
            "precision_human": precision_score(y_true, y_pred, pos_label=0),
            "recall_human": recall_score(y_true, y_pred, pos_label=0),
            "f1_human": f1_score(y_true, y_pred, pos_label=0),
        }
        results.append(result)

    return pd.DataFrame(results)


def evaluate_gptzero(texts, labels, threshold=40.0):
    texts = np.array(texts)
    labels = np.array(labels)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    y_true, y_pred, perplexities = [], [], []

    for i, text in enumerate(eval_texts):
        print(f"Scoring {i+1}/{len(eval_texts)}")
        pred, ppl = classify_text(text, threshold=threshold)
        y_true.append(eval_labels[i])
        y_pred.append(pred)
        perplexities.append(ppl)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Human", "AI"]))

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_AI": precision_score(y_true, y_pred, pos_label=1),
        "recall_AI": recall_score(y_true, y_pred, pos_label=1),
        "f1_AI": f1_score(y_true, y_pred, pos_label=1),
        "precision_human": precision_score(y_true, y_pred, pos_label=0),
        "recall_human": recall_score(y_true, y_pred, pos_label=0),
        "f1_human": f1_score(y_true, y_pred, pos_label=0),
    }