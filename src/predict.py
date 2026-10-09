"""
Module: src/predict.py
Purpose: Standalone inference module for Spam Message Detection.
Loads saved TF-IDF vectorizer and trained model to make real-time predictions.
"""

import os
import re
import joblib


def clean_text(text: str) -> str:
    """
    Standardize text identically to the training stage:
      - Lowercase
      - Replace URLs with 'url' token
      - Replace phone numbers / long digit sequences with 'num' token
      - Remove non-alphanumeric characters
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


class SpamPredictor:
    """
    SpamPredictor encapsulates loading the serialized model & vectorizer
    and running single or batch inferences.
    """

    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "spam_model.pkl")
        if vectorizer_path is None:
            vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. Please run 'python src/train.py' first."
            )
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer file not found at: {vectorizer_path}. Please run 'python src/train.py' first."
            )

        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, raw_message: str) -> dict:
        """
        Takes raw string input, cleans it, vectorizes it, and outputs:
          - prediction: 'SPAM' or 'NOT SPAM'
          - spam_probability: float (0.0 to 1.0)
          - ham_probability: float (0.0 to 1.0)
          - confidence: float (confidence of predicted class)
          - cleaned_text: string after cleaning
        """
        if not raw_message or not raw_message.strip():
            return {
                "error": "Message is empty. Please provide a valid text message."
            }

        cleaned = clean_text(raw_message)

        # Check if text became empty after removing punctuation/spaces
        if not cleaned:
            return {
                "prediction": "NOT SPAM",
                "label": "ham",
                "spam_probability": 0.0,
                "ham_probability": 1.0,
                "confidence": 1.0,
                "cleaned_text": ""
            }

        # Transform using pre-fitted TF-IDF vectorizer (DO NOT FIT AGAIN)
        features = self.vectorizer.transform([cleaned])

        # Predict class (0 = ham, 1 = spam)
        pred_class = int(self.model.predict(features)[0])

        # Predict probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            ham_prob = float(probs[0])
            spam_prob = float(probs[1])
        else:
            # Fallback if model doesn't support probabilities directly
            spam_prob = 1.0 if pred_class == 1 else 0.0
            ham_prob = 1.0 - spam_prob

        is_spam = pred_class == 1
        return {
            "prediction": "SPAM" if is_spam else "NOT SPAM",
            "label": "spam" if is_spam else "ham",
            "is_spam": is_spam,
            "spam_probability": spam_prob,
            "ham_probability": ham_prob,
            "confidence": spam_prob if is_spam else ham_prob,
            "cleaned_text": cleaned
        }


if __name__ == "__main__":
    predictor = SpamPredictor()

    test_samples = [
        "Congratulations! You won a guaranteed £1000 cash prize or holiday. Call 09050000321 now!",
        "Hi Mom, I will be late for dinner today. Please don't wait for me.",
        "URGENT: Your account balance is low. Claim your free reward points now at http://bit.ly/spam",
        "Can you please send me the class notes from yesterday's lecture?",
        "Free entry in 2 a weekly competition to win FA Cup final tickets! Text FA to 87121."
    ]

    print("=" * 70)
    print("RUNNING INFERENCE TESTS ON SAMPLE MESSAGES")
    print("=" * 70)

    for msg in test_samples:
        result = predictor.predict(msg)
        print(f"\nMessage: '{msg}'")
        print(f"Prediction:        {result['prediction']}")
        print(f"Spam Probability:  {result['spam_probability'] * 100:.2f}%")
        print(f"Ham Probability:   {result['ham_probability'] * 100:.2f}%")
        print(f"Cleaned Text:      '{result['cleaned_text']}'")
        print("-" * 50)
