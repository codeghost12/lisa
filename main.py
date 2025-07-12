import pandas as pd
import numpy as np
from feature_extraction import extract_features_for_all
from sota.bert import evaluate_bert, evaluate_bert_on_ratios
from sota.detectgpt import evaluate_detectgpt_on_ratios, evaluate_detectgpt
from sota.gptzero import evaluate_gptzero, evaluate_gptzero_on_ratios
from sota.roberta import evaluate_roberta_on_ratios, evaluate_roberta
from sklearn.model_selection import train_test_split
from lisa.lisa_models import train_and_evaluate_lisa_models, train_and_evaluate_lisa_models_ratios

# Load dataset
print("Loading dataset")
data = pd.read_csv('data/xsum.csv')
texts = data['text'].tolist()
labels = data['label'].tolist()

# Extract advanced features
print("Extracting features")
full_features, tfidf_vectorizer = extract_features_for_all(texts)

# Get subsets (10k and 20k)
print("Creating subsets for dataset")
subset_10k_features = full_features[:10_000]
subset_10k_labels = labels[:10_000]
subset_10k_texts = texts[:10_000]

subset_20k_features = full_features[:20_000]
subset_20k_labels = labels[:20_000]
subset_20k_texts = texts[:20_000]

def evaluate_sota_full(texts, labels):
    print("====Evaluating without ratios (full_dataset)===")
    print("\tBERT:")
    print(evaluate_bert(texts, labels))
    print("\tRoBERTa:")
    print(evaluate_roberta(texts, labels))
    print("\tGPTZero:")
    print(evaluate_gptzero(texts, labels))
    print("\tDetectGPT:")
    print(evaluate_detectgpt(texts, labels))


def evaluate_sota_ratios(texts, labels):
    print("====Evaluating with ratios===")
    print("\tBERT:")
    print(evaluate_bert_on_ratios(texts, labels))
    print("\tRoBERTa:")
    print(evaluate_roberta_on_ratios(texts, labels))
    print("\tGPTZero:")
    print(evaluate_gptzero_on_ratios(texts, labels))
    print("\tDetectGPT:")
    print(evaluate_detectgpt_on_ratios(texts, labels))


def evaluate_lisa_full(features, labels):
    lisa_results_full = train_and_evaluate_lisa_models(features, labels)

    # Save results or display
    for model_name, report in lisa_results_full.items():
        print(f"\nResults for {model_name}:")
        print(report)


def evaluate_lisa_ratios(features, labels):
    lisa_results = train_and_evaluate_lisa_models_ratios(features, labels)
    for model_name, report in lisa_results.items():
        print(f"\nResults for {model_name}:")
        print(report) 


# full dataset (XSum)
print("Evaluating without ratios (full dataset)")
evaluate_sota_full(texts, labels)
evaluate_lisa_full(full_features, labels)

print("Evaluating without ratios (10k)")
evaluate_sota_full(subset_10k_texts, subset_10k_labels)
evaluate_lisa_full(subset_10k_features, subset_10k_labels)

print("Evaluating without ratios (20k)")
evaluate_sota_full(subset_20k_texts, subset_20k_labels)
evaluate_lisa_full(subset_20k_features, subset_20k_labels)

print("Evaluating with ratios (full dataset)")
evaluate_sota_ratios(texts, labels)
evaluate_lisa_ratios(full_features, labels)

print("Evaluating with ratios (10k)")
evaluate_sota_ratios(subset_10k_texts, subset_10k_labels)
evaluate_lisa_ratios(subset_10k_features, subset_10k_labels)

print("Evaluating with ratios (20k)")
evaluate_sota_ratios(subset_20k_texts, subset_20k_labels)
evaluate_lisa_ratios(subset_20k_features, subset_20k_labels)