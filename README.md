# 🛡️ Spam Message Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=kvsruchitha-commits/sms-spam-detection-ml&branch=main&mainModule=app.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Machine Learning system that accurately classifies incoming SMS messages as **SPAM** or **NOT SPAM (HAM)** in real time, accompanied by confidence probability scores and an interactive web interface.

---

## 📌 Project Overview
Unsolicited spam messages, phishing lures, and deceptive marketing messages are pervasive security risks. This project applies statistical NLP and supervised machine learning to automatically analyze message content and protect users from fraudulent messages. 

The project follows industry-standard data science practices:
- Data exploration and deduplication (preventing train-test contamination)
- Domain-specific text preprocessing (URL and phone number tokenization)
- Stratified train/test splitting to handle class imbalance
- Feature extraction using **Term Frequency-Inverse Document Frequency (TF-IDF)**
- Benchmarking three distinct classification algorithms:
  - **Multinomial Naive Bayes**
  - **Logistic Regression**
  - **Calibrated Linear Support Vector Machine (Linear SVM)**
- Complete evaluation using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices
- Real-time web deployment powered by **Streamlit**

---

## 🎯 Problem Statement
Given an arbitrary text message, classify whether it belongs to:
1. `ham` (0): Legitimate personal or transactional communication.
2. `spam` (1): Unsolicited marketing, lottery scam, phishing attempt, or fraudulent broadcast.

In spam detection, **false positives (legitimate messages classified as spam) carry severe consequences** (e.g., missing critical bank alerts, job notifications, or family communications). Therefore, our evaluation prioritizes **high Precision** while simultaneously maintaining strong **Recall**.

---

## 🎯 Objectives
- Build a genuine ML classification model without hardcoded keywords or artificial heuristics.
- Conduct exploratory data analysis to uncover linguistic differences between spam and ham.
- Implement a leak-free NLP vectorization pipeline (fitting TF-IDF exclusively on training data).
- Train and rigorously compare multiple machine learning algorithms.
- Deploy the optimal model as an accessible, responsive web application using Streamlit.
- Provide comprehensive documentation suitable for technical portfolio review and academic viva.

---

## 📊 Dataset
The project utilizes the benchmark **UCI SMS Spam Collection** dataset:
- **Source:** [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- **Raw Count:** 5,572 tab-separated messages.
- **Deduplication:** 403 duplicate messages removed to ensure statistical independence between splits.
- **Clean Unique Samples:** 5,169 messages.
  - **Ham (Legitimate):** 4,516 messages (87.37%)
  - **Spam:** 653 messages (12.63%)
- **Data Imbalance:** Approximately 7:1 ratio of legitimate to spam messages, reflecting realistic real-world distributions.

---

## 🛠️ Technologies Used
| Component | Technology | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core programming runtime |
| **Data Manipulation** | `pandas`, `numpy` | Tabular data manipulation, array processing |
| **Data Visualization** | `matplotlib`, `seaborn` | Distribution analysis, confusion matrix plots |
| **Machine Learning** | `scikit-learn` | TF-IDF vectorization, classifiers, evaluation metrics |
| **Model Persistence** | `joblib` | Fast disk serialization of trained weights & vocabulary |
| **Web Dashboard** | `Streamlit` | Interactive, lightweight web UI |
| **Version Control** | `Git`, `GitHub` | Source code management and collaboration |

---

## 🔄 Machine Learning Workflow
```
Raw SMS Dataset (UCI)
       │
       ▼
Data Inspection & Deduplication (5,169 unique samples)
       │
       ▼
Text Normalization (Lowercasing, URL/Phone tokens, Regex cleaning)
       │
       ▼
Stratified Train/Test Split (80% Train : 20% Test, stratify=y)
       │
       ▼
TF-IDF Vectorization (fit on Train ONLY, transform Test)
       │
       ▼
Model Benchmarking (Naive Bayes, Logistic Regression, Linear SVM)
       │
       ▼
Rigorous Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
       │
       ▼
Model Selection (Calibrated Linear SVM: 98.65% Acc, 97.56% Prec, 94.49% F1)
       │
       ▼
Serialization (joblib dump: spam_model.pkl & tfidf_vectorizer.pkl)
       │
       ▼
Streamlit Web Application & Real-Time Inference
```

---

## 🧹 Data Preprocessing
Raw SMS messages often contain informal punctuation, URLs, telephone numbers, and capitalization. Our text normalization pipeline implements:
1. **Lowercasing:** Standardizes case variations (e.g., `FREE` vs `free`).
2. **URL Tokenization:** Replaces web addresses with a unified `url` token using regular expression `http\S+|www\.\S+`.
3. **Phone/Number Tokenization:** Replaces long digit sequences (5+ digits) with a `num` token using `\b\d{5,}\b`.
4. **Punctuation Removal:** Strips non-alphanumeric characters while preserving word separations.
5. **Whitespace Normalization:** Trims redundant spaces and tabs.
6. **Stop Words Filtering:** Standard English stop words (`the`, `is`, `at`, `which`) are filtered inside TF-IDF vectorization to focus on content-bearing vocabulary.

---

## 🔢 TF-IDF Feature Extraction
Machine learning models operate on numerical vectors, not strings. We apply **Term Frequency-Inverse Document Frequency (TF-IDF)**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

- **Term Frequency (TF):** Measures how frequently a token $t$ appears within a specific message $d$.
- **Inverse Document Frequency (IDF):** Penalizes tokens that appear everywhere across all documents, while rewarding tokens that uniquely characterize specific messages (e.g., `winner`, `urgent`, `prize`, `claim`).
- **Feature Vocabulary:** Limited to the top 3,000 discriminative n-grams.
- **Data Leakage Prevention:** The vectorizer is **fitted only on `X_train`**; `X_test` and new incoming messages are strictly **transformed** using the learned parameters.

---

## 📊 Model Evaluation & Comparison
All models were tested on the unseen stratified test set (1,034 messages: 903 Ham, 131 Spam):

| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score | FP (Legit $\rightarrow$ Spam) | FN (Spam Missed) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | 97.87% | **99.10%** | 83.97% | 90.91% | **1** | 21 |
| **Logistic Regression** | 97.20% | 98.11% | 79.39% | 87.76% | 2 | 27 |
| **Calibrated Linear SVM (Selected)** | **98.65%** | **97.56%** | **91.60%** | **94.49%** | **3** | **11** |

### Detailed Confusion Matrix for Final Selected Model (Calibrated Linear SVM)
```
                  Predicted HAM    Predicted SPAM
Actual HAM:            900               3        (False Positives = 3)
Actual SPAM:            11             120        (True Positives = 120, Missed = 11)
```

---

## 🏆 Final Model Selection Justification
**Calibrated Linear SVM** was chosen for deployment:
1. **Superior Overall F1-Score (94.49%):** It captures **91.60% of all spam messages** (120 out of 131) while Naive Bayes misses 21 and Logistic Regression misses 27.
2. **High Precision (97.56%):** Out of 903 legitimate messages, only 3 were flagged, keeping inbox disruption negligible.
3. **Calibrated Probabilities:** By wrapping `LinearSVC` with Platt scaling (`CalibratedClassifierCV`), the system computes smooth, calibrated probability scores (0% to 100%) rather than arbitrary decision margins.

---

## 📁 Project Structure
```
SPAM DECTION PROJECT/
├── data/
│   └── SMSSpamCollection               # Tab-separated dataset from UCI
├── models/
│   ├── spam_model.pkl                  # Serialized Calibrated Linear SVM model
│   └── tfidf_vectorizer.pkl            # Serialized fitted TF-IDF vectorizer
├── notebooks/
│   └── spam_detection_analysis.ipynb   # Interactive analysis, EDA & visual charts
├── src/
│   ├── __init__.py                     # Module initializer
│   ├── train.py                        # Full training, evaluation & artifact export script
│   └── predict.py                      # Standalone prediction and inference module
├── app.py                              # Streamlit interactive web application
├── requirements.txt                    # Project dependency specification
├── README.md                           # Comprehensive documentation
└── .gitignore                          # Excludes venv, pycache, temporary artifacts
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kvsruchitha-commits/sms-spam-detection-ml.git
cd sms-spam-detection-ml
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
```bash
python src/train.py
```
*This step loads the data, computes evaluation metrics, and generates the `.pkl` files inside `models/`.*

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Example Test Predictions

| Input Message | Predicted Class | Spam Probability | Legit Probability |
| :--- | :---: | :---: | :---: |
| *"Congratulations! You won a guaranteed £1000 cash prize or holiday. Call 09050000321 now!"* | 🚨 **SPAM** | **99.99%** | 0.01% |
| *"Hi Mom, I will be late for dinner today. Please don't wait for me."* | ✅ **NOT SPAM** | 0.20% | **99.80%** |
| *"URGENT: Your account balance is low. Claim your free reward points now at http://bit.ly/spam"* | 🚨 **SPAM** | **99.85%** | 0.15% |
| *"Can you please send me the class notes from yesterday's lecture?"* | ✅ **NOT SPAM** | 0.40% | **99.60%** |

---

## ⚠️ Limitations
- **English-Centric Vocabulary:** The TF-IDF stop words and tokenizers are tailored for English text; multilingual or transliterated SMS (e.g., Hinglish) may exhibit reduced accuracy without specialized language models.
- **Evasion / Obfuscation:** Attackers who intentionally disguise trigger words (e.g., `w!nn3r`, `fr33`) might slip past bag-of-words representations unless character-level n-grams or subword tokenization are used.
- **Contextual Limitations:** TF-IDF treats n-grams statistically and lacks deep semantic context compared to Transformer architectures (such as DistilBERT).

---

## 🔮 Future Enhancements
- Integrate subword embeddings (FastText) or lightweight Transformers (DistilBERT / TinyBERT).
- Implement user feedback reporting inside the Streamlit app to enable continuous learning (active learning loop).
- Add support for multilingual spam filtering using language-agnostic embeddings.
- Package as a lightweight REST API using FastAPI alongside a Docker container.

---

## 👨‍💻 Author & Acknowledgements
- **Author:** [kvsruchitha-commits](https://github.com/kvsruchitha-commits)
- **Repository:** [sms-spam-detection-ml](https://github.com/kvsruchitha-commits/sms-spam-detection-ml)
- **Dataset Credits:** UCI Machine Learning Repository (Almeida, T.A., Hidalgo, J.M.G., Yamakami, A.)
- **License:** MIT License
