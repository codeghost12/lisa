import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"

def evaluate_roberta_on_ratios(
    texts, labels, ratios=[(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
):
    results = []
    texts = np.array(texts)
    labels = np.array(labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        accuracy = accuracy_score(labels, preds)
        report = classification_report(labels, preds, output_dict=True)

        print("Classification Report:\n", classification_report(labels, preds))
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

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

        # Stratified train-test split
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            selected_texts, selected_labels, test_size=0.2, stratify=selected_labels, random_state=42
        )

        # Print label distribution
        print(f"Training samples: {len(train_texts)}")
        print(f"  -> Human: {np.sum(train_labels == 0)}")
        print(f"  -> AI: {np.sum(train_labels == 1)}")
        print(f"Evaluation samples: {len(eval_texts)}")
        print(f"  -> Human: {np.sum(eval_labels == 0)}")
        print(f"  -> AI: {np.sum(eval_labels == 1)}")

        # Convert to Hugging Face Datasets
        train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
        eval_dataset = Dataset.from_dict({'text': eval_texts.tolist(), 'label': eval_labels.tolist()})

        # Tokenize
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Load model
        model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            eval_strategy="epoch",
            learning_rate=2e-5,
            warmup_steps=0,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="no"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        eval_result = trainer.evaluate()

        result = {'human_ratio': human_ratio, 'ai_ratio': ai_ratio}
        result.update(eval_result)
        results.append(result)

    return pd.DataFrame(results)


def evaluate_roberta(texts, labels):
    texts = np.array(texts)
    labels = np.array(labels)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        accuracy = accuracy_score(labels, preds)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
    eval_dataset = Dataset.from_dict({'text': eval_texts.tolist(), 'label': eval_labels.tolist()})

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results-simple-roberta',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir='./logs-simple-roberta',
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()

    return eval_result
