# lisa_methods.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np

def train_and_evaluate_lisa_models(features, labels, test_size=0.2, random_state=42):
    results = {}
    # Split inside the function
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    # Random Forest
    rfc = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    results["RandomForest"] = classification_report(y_test, y_pred, output_dict=True)

    # MLP
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=random_state)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    results["MLP"] = classification_report(y_test, y_pred, output_dict=True)

    # SVM
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results["SVM"] = classification_report(y_test, y_pred, output_dict=True)

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    results["XGBoost"] = classification_report(y_test, y_pred, output_dict=True)

    return results


def train_and_evaluate_lisa_models_ratios(features, labels, ratios=[(0.7,0.3), (0.5,0.5), (0.3,0.7)]):
    results_ratios = {}
    labels_array = np.array(labels)
    ai_indices = np.where(labels_array == 1)[0]
    human_indices = np.where(labels_array == 0)[0]

    for ai_ratio, human_ratio in ratios:
        print(f"\nEvaluating LISA models with AI:Human ratio = {ai_ratio}:{human_ratio}")

        ai_sample_size = int(min(len(ai_indices) / ai_ratio, len(human_indices) / human_ratio) * ai_ratio)
        human_sample_size = int(min(len(ai_indices) / ai_ratio, len(human_indices) / human_ratio) * human_ratio)

        rng = np.random.default_rng(seed=42)
        ai_sample_indices = rng.choice(ai_indices, size=ai_sample_size, replace=False)
        human_sample_indices = rng.choice(human_indices, size=human_sample_size, replace=False)

        selected_indices = np.concatenate([ai_sample_indices, human_sample_indices])
        rng.shuffle(selected_indices)

        custom_features = features[selected_indices]
        custom_labels = labels_array[selected_indices]

        X_train, X_test, y_train, y_test = train_test_split(
            custom_features, custom_labels, test_size=0.2, stratify=custom_labels, random_state=42
        )

        results = train_and_evaluate_lisa_models(X_train, X_test, y_train, y_test)
        results_ratios[f"AI_{int(ai_ratio*100)}_Human_{int(human_ratio*100)}"] = results

        # Print class distribution
        final_counts = Counter(custom_labels)
        print("Final class distribution:")
        for label, count in final_counts.items():
            label_name = "AI" if label == 1 else "Human"
            print(f"{label_name}: {count}")

    return results_ratios