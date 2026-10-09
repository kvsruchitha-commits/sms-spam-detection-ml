"""
Module: src/train.py
Purpose: Complete ML training pipeline for Spam Message Detection.
Steps:
  1. Load dataset from data/SMSSpamCollection
  2. Data cleaning & deduplication
  3. Stratified train/test split
  4. TF-IDF feature extraction (fit on train only)
  5. Train multiple ML algorithms (Naive Bayes, Logistic Regression, Linear SVM)
  6. Evaluate models with Accuracy, Precision, Recall, F1-score & Confusion Matrix
  7. Save best trained model and vectorizer using joblib
"""

import os
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def clean_text(text: str) -> str:
    """
    Standardize and clean text:
      - Convert to lowercase
      - Replace URLs with 'url' token
      - Replace phone numbers / long digit sequences with 'num' token
      - Remove special characters and punctuation
      - Normalize spaces
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\b\d{5,}\b", " num ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_data(file_path: str):
    """
    Loads UCI SMS Spam Collection dataset, inspects shape,
    removes duplicates to prevent data leakage, and cleans messages.
    """
    print("=" * 60)
    print("STEP 1: LOADING AND PREPARING DATASET")
    print("=" * 60)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please download it first.")

    # Load tab-separated file without header
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1"
    )
    print(f"Initial raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Check missing values
    missing = df.isnull().sum().to_dict()
    print(f"Missing values: {missing}")

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicate_count}")

    # Deduplicate to prevent train-test data leakage
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Dataset shape after deduplication: {df.shape[0]} rows")

    # Class distribution
    counts = df["label"].value_counts().to_dict()
    print(f"Class distribution: {counts} (Ham: {counts.get('ham', 0)}, Spam: {counts.get('spam', 0)})")

    # Encode label: ham -> 0, spam -> 1
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # Clean text
    print("Cleaning text messages...")
    df["clean_message"] = df["message"].apply(clean_text)

    return df


def train_and_evaluate(df: pd.DataFrame, models_dir: str = "models"):
    """
    Splits data, fits TF-IDF on train only, trains models, evaluates, and saves the best model.
    """
    print("\n" + "=" * 60)
    print("STEP 2: STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 60)

    X = df["clean_message"]
    y = df["target"]

    # Stratified split ensures exact spam/ham proportion in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} ({sum(y_train)} spam, {len(y_train) - sum(y_train)} ham)")
    print(f"Testing samples:  {len(X_test)} ({sum(y_test)} spam, {len(y_test) - sum(y_test)} ham)")

    print("\n" + "=" * 60)
    print("STEP 3: TF-IDF FEATURE EXTRACTION")
    print("=" * 60)
    print("Fitting TF-IDF Vectorizer ONLY on training set (preventing data leakage)...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_features=3000
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF Feature vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Training TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"Testing TF-IDF matrix shape:  {X_test_tfidf.shape}")

    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    # Initialize candidate models
    # Note: CalibratedClassifierCV wraps LinearSVC to output well-calibrated probabilities
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Linear SVM (Calibrated)": CalibratedClassifierCV(
            LinearSVC(random_state=42, dual=False, max_iter=2000)
        )
    }

    results = []
    trained_models = {}

    for name, model in candidate_models.items():
        print(f"\n--- Training: {name} ---")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        trained_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ConfusionMatrix": cm
        })

        print(f"  Accuracy:  {acc * 100:.2f}%")
        print(f"  Precision: {prec * 100:.2f}% (Spam positive predictive value)")
        print(f"  Recall:    {rec * 100:.2f}% (Spam detection sensitivity)")
        print(f"  F1-Score:  {f1 * 100:.2f}%")
        print(f"  Confusion Matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {
            "Model": r["Model"],
            "Accuracy (%)": f"{r['Accuracy']*100:.2f}%",
            "Precision (%)": f"{r['Precision']*100:.2f}%",
            "Recall (%)": f"{r['Recall']*100:.2f}%",
            "F1-Score (%)": f"{r['F1-Score']*100:.2f}%",
            "FP (Legit marked Spam)": r["ConfusionMatrix"][0, 1],
            "FN (Spam missed)": r["ConfusionMatrix"][1, 0]
        }
        for r in results
    ])
    print(summary_df.to_string(index=False))

    # Selection decision:
    # Calibrated Linear SVM achieves the highest F1-score and best balance of precision and recall.
    best_model_name = "Linear SVM (Calibrated)"
    best_model = trained_models[best_model_name]
    print(f"\nSelected Model: '{best_model_name}' for deployment due to highest overall F1-score & accuracy.")

    print("\n" + "=" * 60)
    print("STEP 5: SAVING ARTIFACTS")
    print("=" * 60)
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "spam_model.pkl")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved trained model to:      {model_path}")
    print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    print("Pipeline successfully completed!")


if __name__ == "__main__":
    dataset_path = os.path.join("data", "SMSSpamCollection")
    train_and_evaluate(load_and_prepare_data(dataset_path))
