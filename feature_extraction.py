import numpy as np
import nltk
import spacy
import torch
import re
import math
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertModel, BertForNextSentencePrediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Setup
nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# ------------------------ Feature Collection Helper Functions ------------------------ #
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    return " ".join([token for token in tokens if token.isalpha() and token not in stop_words])

def get_bert_embeddings(text):
    if not text.strip():
        return np.zeros(bert_model.config.hidden_size)
    bert_tokens = bert_tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        bert_outputs = bert_model(**bert_tokens)
        return bert_outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def kl_divergence(p, q):
    epsilon = 1e-10
    return np.sum(np.where(p != 0, p * np.log((p + epsilon) / (q + epsilon)), 0))

def get_topic_distributions(paragraphs, num_topics=5):
    non_empty = [p for p in paragraphs if p.strip()]
    if not non_empty:
        return np.zeros((len(paragraphs), num_topics))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(non_empty)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    return lda.transform(vectorizer.transform(paragraphs))

def compute_repetition_rate(text, n=3):
    words = word_tokenize(text.lower())
    if not words:
        return 0, 0
    word_counts = Counter(words)
    word_rep = sum(c - 1 for c in word_counts.values() if c > 1) / len(words)
    n_grams = list(ngrams(words, n))
    if not n_grams:
        return word_rep, 0
    ngram_counts = Counter(n_grams)
    ngram_rep = sum(c - 1 for c in ngram_counts.values() if c > 1) / len(n_grams)
    return word_rep, ngram_rep

def compute_entropy(text):
    words = text.lower().split()
    if not words:
        return 0
    freq = Counter(words)
    total = len(words)
    return -sum((c/total) * math.log2(c/total) for c in freq.values() if c > 0)

def compute_tfidf(texts, vectorizer=None):
    if vectorizer is None:
        vectorizer = CountVectorizer()
        vectorizer.fit([t for t in texts if t.strip()])
    return vectorizer.transform(texts).toarray(), vectorizer

def compute_pos_distribution(text):
    doc = nlp(text)
    pos_counts = Counter(token.pos_ for token in doc)
    total = sum(pos_counts.values())
    pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CCONJ', 'INTJ', 'NUM']
    return [pos_counts.get(tag, 0) / total if total else 0 for tag in pos_tags]

def compute_avg_parse_depth(text):
    doc = nlp(text)
    if not doc:
        return 0
    depths = [len(list(token.ancestors)) for token in doc]
    return sum(depths) / len(depths) if depths else 0

def compute_lexical_diversity(text):
    words = word_tokenize(text.lower())
    return len(set(words)) / len(words) if words else 0

def compute_burstiness(text):
    words = word_tokenize(text.lower())
    freqs = list(Counter(words).values())
    return np.var(freqs) if freqs else 0

def compute_sentence_nsp_score(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return 0.5
    scores = []
    for i in range(len(sentences) - 1):
        inputs = bert_tokenizer(sentences[i], sentences[i+1], return_tensors='pt')
        with torch.no_grad():
            logits = nsp_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)
            scores.append(probs[0][0].item())  # Next sentence score
    return sum(scores) / len(scores)

# ------------------------ Main Feature Extraction ------------------------ #
def extract_features_for_all(texts, tfidf_vectorizer=None, preprocessed_texts=None):
    print("Starting advanced feature extraction...")
    features = []

    if preprocessed_texts is None:
        preprocessed_texts = [preprocess_text(t) for t in texts]

    tfidf_array, tfidf_vectorizer = compute_tfidf(preprocessed_texts, tfidf_vectorizer)
    print("TF-IDF computed.")

    for idx, text in enumerate(preprocessed_texts):
        if not text.strip():
            vector = [0] * (19 + tfidf_array.shape[1])
        else:
            bert_vec = get_bert_embeddings(text)
            kl_vec = get_topic_distributions([text])
            word_rep, ngram_rep = compute_repetition_rate(text)
            entropy_val = compute_entropy(text)
            tfidf_vec = tfidf_array[idx]

            # New features
            pos_dist = compute_pos_distribution(text)
            parse_depth = compute_avg_parse_depth(text)
            lex_div = compute_lexical_diversity(text)
            burst = compute_burstiness(text)
            nsp_score = compute_sentence_nsp_score(text)

            vector = [
                float(bert_vec.mean()),
                float(kl_vec.mean()),
                float(word_rep),
                float(ngram_rep),
                float(entropy_val),
                float(parse_depth),
                float(lex_div),
                float(burst),
                float(nsp_score),
                *pos_dist,
                *tfidf_vec.tolist()
            ]
        features.append(vector)
        print(f"Processed {idx+1}/{len(texts)}")

    return np.array(features), tfidf_vectorizer