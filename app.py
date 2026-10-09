"""
Streamlit Web Application: Spam Message Detection
Author: Senior ML Engineer & Student
Description: Professional, interactive web dashboard for real-time SMS spam detection
             using a trained Calibrated Linear Support Vector Machine and TF-IDF NLP pipeline.
"""

import os
import streamlit as st
from src.predict import SpamPredictor

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SMS Spam Detector | Machine Learning",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-spam {
        background-color: #FEF2F2;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    .card-ham {
        background-color: #ECFDF5;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Model Resource (Cached for speed)
# ---------------------------------------------------------
@st.cache_resource
def load_predictor():
    try:
        return SpamPredictor()
    except Exception as e:
        return None, str(e)


predictor_result = load_predictor()
if isinstance(predictor_result, tuple):
    predictor = None
    load_error = predictor_result[1]
else:
    predictor = predictor_result
    load_error = None


# ---------------------------------------------------------
# Sidebar Information
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=70)
    st.title("Project Overview")
    st.markdown("""
    **Model:** Calibrated Linear SVM  
    **Vectorization:** TF-IDF (3,000 features)  
    **Training Dataset:** UCI SMS Spam Collection  
    **Accuracy:** `98.65%`  
    **Precision:** `97.56%`  
    **Recall:** `91.60%`  
    **F1-Score:** `94.49%`  
    """)

    st.markdown("---")
    st.subheader("💡 Why TF-IDF + SVM?")
    st.markdown("""
    - **TF-IDF** assigns importance scores to words that specifically discriminate between spam and normal conversations.
    - **Linear SVM** creates a maximum-margin hyperplane separating spam from legitimate messages in high-dimensional text space.
    - **Probability Calibration** transforms distance-from-margin into a calibrated probability between 0% and 100%.
    """)

    st.markdown("---")
    st.caption("Built with Python, Scikit-Learn & Streamlit")


# ---------------------------------------------------------
# Main Page Layout
# ---------------------------------------------------------
st.markdown('<div class="main-header">🛡️ SMS Spam Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time Machine Learning text classification powered by Natural Language Processing.</div>',
    unsafe_allow_html=True
)

if load_error or predictor is None:
    st.error(f"⚠️ Model files could not be loaded: {load_error}")
    st.info("👉 Please run `python src/train.py` in your terminal to train and save the model artifacts.")
    st.stop()


# ---------------------------------------------------------
# Sample Test Messages (Beginner-friendly Quick Fill)
# ---------------------------------------------------------
st.markdown("### 📝 Quick Test Examples")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

if "user_text" not in st.session_state:
    st.session_state["user_text"] = ""

with col_s1:
    if st.button("🎁 Winner Spam", use_container_width=True):
        st.session_state["user_text"] = "Congratulations! You won a £1000 cash prize or holiday. Call 09050000321 now to claim!"

with col_s2:
    if st.button("🚨 Urgent Bank Spam", use_container_width=True):
        st.session_state["user_text"] = "URGENT: Your account balance is low. Claim your free reward points now at http://secure-banking.com"

with col_s3:
    if st.button("👋 Meeting / Ham", use_container_width=True):
        st.session_state["user_text"] = "Hey, are we still meeting for coffee at 4 PM in the library cafeteria?"

with col_s4:
    if st.button("📚 Class Notes / Ham", use_container_width=True):
        st.session_state["user_text"] = "Hi, can you please send me today's lecture notes and assignment guidelines?"


# ---------------------------------------------------------
# Text Input Area
# ---------------------------------------------------------
st.markdown("### 💬 Enter Message Text")
user_input = st.text_area(
    label="Type or paste any SMS message below:",
    value=st.session_state["user_text"],
    height=120,
    placeholder="e.g. Free entry in 2 a weekly competition to win tickets! Text WIN to 87121."
)

col_btn1, col_btn2, col_info = st.columns([1.5, 1, 3])
with col_btn1:
    check_btn = st.button("🔍 Check Message", type="primary", use_container_width=True)
with col_btn2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state["user_text"] = ""
    st.rerun()

# Text statistics
if user_input.strip():
    char_count = len(user_input)
    word_count = len(user_input.split())
    with col_info:
        st.caption(f"📊 Message Length: **{char_count}** characters | **{word_count}** words")


# ---------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------
if check_btn:
    if not user_input or not user_input.strip():
        st.warning("⚠️ Please enter a text message to classify.")
    else:
        with st.spinner("Analyzing message with Natural Language Processing..."):
            result = predictor.predict(user_input)

        if "error" in result:
            st.error(result["error"])
        else:
            is_spam = result["is_spam"]
            spam_pct = result["spam_probability"] * 100
            ham_pct = result["ham_probability"] * 100

            st.markdown("---")
            st.subheader("🎯 Classification Result")

            col_res, col_chart = st.columns([1.5, 1])

            with col_res:
                if is_spam:
                    st.markdown(f"""
                    <div class="card-spam">
                        <div class="result-title" style="color: #DC2626;">🚨 SPAM DETECTED</div>
                        <p style="color: #7F1D1D; font-size: 1.05rem;">
                            This message displays characteristics commonly associated with spam, promotional marketing, phishing, or lottery scams.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card-ham">
                        <div class="result-title" style="color: #059669;">✅ NOT SPAM (HAM)</div>
                        <p style="color: #065F46; font-size: 1.05rem;">
                            This message appears legitimate and conversational. No strong spam patterns were detected.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.caption(f"**Cleaned Tokens Analysed:** `{result['cleaned_text']}`")

            with col_chart:
                st.markdown("#### Confidence Breakdown")
                st.write(f"**Spam Probability:** `{spam_pct:.2f}%`")
                st.progress(float(result["spam_probability"]))

                st.write(f"**Legitimate (Ham) Probability:** `{ham_pct:.2f}%`")
                st.progress(float(result["ham_probability"]))


# ---------------------------------------------------------
# Educational / Architectural Tabs
# ---------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Evaluation Metrics", "⚙️ How It Works", "🧪 Model Comparison"])

with tab1:
    st.markdown("""
    #### Evaluated on Test Set (1,034 SMS Messages)
    | Metric | Calibrated Linear SVM | Multinomial Naive Bayes | Logistic Regression |
    | :--- | :---: | :---: | :---: |
    | **Accuracy** | **98.65%** | 97.87% | 97.20% |
    | **Precision (Spam)** | **97.56%** | 99.10% | 98.11% |
    | **Recall (Spam)** | **91.60%** | 83.97% | 79.39% |
    | **F1-Score** | **94.49%** | 90.91% | 87.76% |
    | **False Positives (Critical)** | **3** | 1 | 2 |
    """)

with tab2:
    st.markdown("""
    #### The 5-Step Detection Pipeline:
    1. **Text Normalization:** Lowercasing, URL tokenization (`http...` $\\rightarrow$ `url`), phone number replacement (`12345...` $\\rightarrow$ `num`), and punctuation stripping.
    2. **Stop Word Filtering:** Common English function words (*the, is, at*) are filtered to isolate high-signal terms.
    3. **TF-IDF Vectorization:** The message is transformed into a 3,000-dimensional numerical vector weighted by term frequency and inverse document frequency.
    4. **Linear Support Vector Classifier:** Projects features into maximum-margin hyperplane decision boundaries.
    5. **Platt Calibration:** Translates hyperplane geometric distances into calibrated, authentic probabilities.
    """)

with tab3:
    st.markdown("""
    #### Why Precision and Recall both matter in Spam Detection:
    - **False Positive (FP):** A legitimate email/SMS from your bank, boss, or university is marked as Spam. **Cost: VERY HIGH.**
    - **False Negative (FN):** A spam message slips into your inbox. **Cost: Annoying, but lower than missing a job offer.**
    - **Decision:** Multinomial Naive Bayes has slightly higher precision, but misses 21 spam messages. Calibrated Linear SVM catches 120 out of 131 spams (91.60% Recall) while maintaining a stellar 97.56% precision, making it the most balanced production choice.
    """)
