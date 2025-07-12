# LISA: Linguistic and Statistical Analysis for AI-Generated Text Detection

This repository contains the official implementation of **LISA**, a feature-based framework for detecting AI-generated text (AGT) using linguistically and statistically grounded features.

> **Paper:** *A Feature-based Linguistic and Statistical Framework for AI-Generated Text Detection*  
> **Conference:** ICMLA 2025  

---

## 🔍 Overview

LISA (Linguistic and Statistical Analysis) is an interpretable, efficient framework designed to distinguish between AI-generated and human-written text. By using handcrafted linguistic and statistical features—such as lexical diversity, repetition rates, entropy, syntactic tree depth, semantic coherence, and TF-IDF vectors—LISA achieves performance on par with or better than leading deep learning-based methods in many cases, while remaining lightweight and transparent.

---

## 🧠 Core Features

- **Interpretable:** Built entirely on transparent, explainable features
- **Efficient:** Fast training and inference using classical ML (RF, SVM, MLP, XGBoost)
- **Robust:** Performs reliably across domain shifts and under class imbalance
- **Competitive:** Matches or outperforms models like RoBERTa, GPTZero, and DetectGPT

---

## 📊 Datasets

LISA was tested on two widely used AGT detection datasets:

- **[XSum](https://huggingface.co/datasets/xsum):** BBC news articles with human-written and AI-generated summaries
- **[HC3](https://github.com/Hello-SimpleAI/HC3):** Human and AI responses to academic and general-purpose questions

AI-generated samples for XSum dataset were created using OpenAI's GPT-3.5 Turbo model.

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/codeghost12/ai-text-detection.git
cd ai-text-detection
pip install -r requirements.txt
```

## 🚀 Usage

- Prepare Your Data
Place your dataset file, if doesn't exist already inside the `data/` folder.

- Run the Full Pipeline
To run the entire experiment (feature extraction, all classical ML models and SOTA baselines on the XSum dataset, just run)
```python main.py```

By default, `main.py` will laod `xsum.csv`, extract features, and evaluate all detection methods (LISA, BERT, RoBERTa, DetectGPT, GPTZero) with both balanced and imbalanced class ratios. All results will be printed to your terminal. 

- To test on a different dataset (i.e. HC3):
    - Change the input file name in `main.py`:
    ```data = pd.readcsv('hc3.csv')```
    - Run the script again
    ```python main.py```