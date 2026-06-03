"""
train_model.py - Intent Classifier Training Script
===================================================

This script trains a machine learning model to classify user input into
predefined intents. It uses TF-IDF vectorization to convert text into
numerical features and Logistic Regression for classification.

Usage:
    python train_model.py

Output:
    - models/intent_classifier.pkl  (trained classifier)
    - models/tfidf_vectorizer.pkl   (fitted TF-IDF vectorizer)
    - models/label_encoder.pkl      (fitted label encoder)
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib


def load_intents(filepath: str) -> dict:
    """Load intents data from a JSON file.

    Args:
        filepath: Path to the intents JSON file.

    Returns:
        Dictionary containing the intents data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_training_data(intents_data: dict) -> pd.DataFrame:
    """Convert intents JSON into a structured pandas DataFrame.

    Each pattern is paired with its corresponding intent tag to create
    labeled training examples.

    Args:
        intents_data: Dictionary with intents, patterns, and responses.

    Returns:
        DataFrame with 'pattern' and 'tag' columns.
    """
    patterns = []
    tags = []

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            patterns.append(pattern.lower())
            tags.append(intent["tag"])

    df = pd.DataFrame({"pattern": patterns, "tag": tags})
    print(f"[INFO] Loaded {len(df)} training examples across {df['tag'].nunique()} intents.")
    return df


def train_classifier(df: pd.DataFrame) -> tuple:
    """Train the intent classification pipeline.

    Pipeline:
        1. TF-IDF Vectorization: Converts text patterns into numerical
           feature vectors based on term frequency-inverse document frequency.
        2. Label Encoding: Converts string labels to numerical format.
        3. Logistic Regression: Trains a multi-class classifier on the
           TF-IDF features.

    Args:
        df: DataFrame with 'pattern' and 'tag' columns.

    Returns:
        Tuple of (classifier, vectorizer, label_encoder).
    """
    # Initialize TF-IDF Vectorizer
    # - ngram_range=(1,2): captures unigrams and bigrams for better context
    # - max_features=500: limits vocabulary size to prevent overfitting
    # - sublinear_tf=True: applies logarithmic scaling for better performance
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=500,
        sublinear_tf=True,
        lowercase=True,
    )

    # Transform text patterns into TF-IDF feature matrix
    X = vectorizer.fit_transform(df["pattern"])
    print(f"[INFO] TF-IDF matrix shape: {X.shape}")

    # Encode intent labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["tag"])
    print(f"[INFO] Classes: {list(label_encoder.classes_)}")

    # Train Logistic Regression classifier
    # - max_iter=1000: ensures convergence for small datasets
    # - C=10: slightly less regularization for better fit on small data
    # - solver='lbfgs': efficient solver for multinomial classification
    classifier = LogisticRegression(
        max_iter=1000,
        C=10,
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(X, y)

    # Evaluate with cross-validation
    scores = cross_val_score(classifier, X, y, cv=min(5, len(df["tag"].unique())), scoring="accuracy")
    print(f"[INFO] Cross-validation accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    return classifier, vectorizer, label_encoder


def save_model(classifier, vectorizer, label_encoder, output_dir: str = "models"):
    """Save trained model components to disk.

    Args:
        classifier: Trained Logistic Regression model.
        vectorizer: Fitted TF-IDF vectorizer.
        label_encoder: Fitted label encoder.
        output_dir: Directory to save model files.
    """
    os.makedirs(output_dir, exist_ok=True)

    joblib.dump(classifier, os.path.join(output_dir, "intent_classifier.pkl"))
    joblib.dump(vectorizer, os.path.join(output_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

    print(f"[INFO] Model saved to '{output_dir}/' directory.")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  AI Chatbot - Intent Classifier Training")
    print("=" * 60)
    print()

    # Step 1: Load training data
    print("[STEP 1] Loading intents data...")
    intents_data = load_intents("intents.json")
    print()

    # Step 2: Prepare training DataFrame
    print("[STEP 2] Preparing training data...")
    df = prepare_training_data(intents_data)
    print()

    # Step 3: Display data summary
    print("[STEP 3] Training data summary:")
    print(df["tag"].value_counts().to_string())
    print()

    # Step 4: Train the model
    print("[STEP 4] Training classifier...")
    classifier, vectorizer, label_encoder = train_classifier(df)
    print()

    # Step 5: Save model artifacts
    print("[STEP 5] Saving model...")
    save_model(classifier, vectorizer, label_encoder)
    print()

    print("=" * 60)
    print("  Training complete! Run 'python chatbot.py' to start chatting.")
    print("=" * 60)


if __name__ == "__main__":
    main()
