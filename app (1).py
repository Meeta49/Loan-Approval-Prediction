import streamlit as st
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

# ── Train & cache a model (runs once) ────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """
    If you have a saved model (loan_model.pkl) drop it next to app.py.
    Otherwise we train a lightweight demo model so the app works standalone.
    """
    model_path = "loan_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    # ── Demo training data (mirrors the Kaggle loan-prediction dataset) ──
    np.random.seed(42)
    n = 600
    X = np.column_stack([
        np.random.choice([0, 1], n),           # Gender (Male=1)
        np.random.choice([0, 1], n),           # Married
        np.random.choice([0, 1, 2, 3], n),     # Dependents
        np.random.choice([0, 1], n),           # Education (Graduate=1)
        np.random.choice([0, 1], n),           # Self_Employed
        np.random.randint(1000, 15000, n),     # ApplicantIncome
        np.random.randint(0, 8000, n),         # CoapplicantIncome
        np.random.randint(50, 700, n),         # LoanAmount
        np.random.choice([6, 12, 36, 60, 84, 120, 180, 240, 300, 360, 480], n),  # Loan_Amount_Term
        np.random.choice([0, 1], n, p=[0.15, 0.85]),   # Credit_History
        np.random.choice([0, 1, 2], n),        # Property_Area (Rural=0,Semi=1,Urban=2)
    ])
    # Simple rule: approved if credit history=1 AND income+coincome > loanAmount*2
    y = ((X[:, 9] == 1) & ((X[:, 5] + X[:, 6]) > X[:, 7] * 2)).astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_or_train_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Loan Approval Prediction")
st.markdown("Fill in the applicant details below to check loan eligibility.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    st.subheader("Financial Details")
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (₹ thousands)", min_value=1, value=150, step=10)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84, 60, 36, 12, 480, 300, 240, 6])
    credit_history = st.selectbox("Credit History", ["Meets Guidelines (1)", "Does Not Meet (0)"])

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()

# ── Encode inputs ─────────────────────────────────────────────────────────────
def encode():
    return np.array([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents],
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        1 if credit_history.startswith("Meets") else 0,
        {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area],
    ]])

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Loan Approval", use_container_width=True, type="primary"):
    features = encode()
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    st.divider()
    if prediction == 1:
        st.success("✅ **Loan Approved!**")
        st.metric("Approval Confidence", f"{probability[1]*100:.1f}%")
        st.balloons()
    else:
        st.error("❌ **Loan Not Approved**")
        st.metric("Rejection Confidence", f"{probability[0]*100:.1f}%")
        st.info("💡 Tip: Improving credit history or reducing loan amount may increase chances of approval.")

    # Feature importance mini-insight
    with st.expander("📊 Key Factors Considered"):
        factors = {
            "Credit History": "High impact — the most critical factor",
            "Applicant Income": f"₹{applicant_income:,}/month",
            "Co-applicant Income": f"₹{coapplicant_income:,}/month",
            "Loan Amount": f"₹{loan_amount}k over {loan_term} months",
            "Property Area": property_area,
        }
        for k, v in factors.items():
            st.write(f"**{k}:** {v}")

st.caption("Model: Random Forest Classifier | Dataset: Loan Prediction (Analytics Vidhya)")
