# ═══════════════════════════════════════════════════════════════
# app.py — Streamlit Customer Churn Predictor
# Run: streamlit run app.py
# ═══════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import pickle

# ─────────────────────────────────────────────
# Load Model Artifacts
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():

    with open("random_forest.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


model, scaler, feature_names = load_model()

# Numeric columns used during training
numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_charges_per_month",
    "charges_tenure_ratio"
]

# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details to estimate churn probability.")

col1, col2, col3 = st.columns(3)

# ───────── Column 1 ─────────
with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=18.0,
        max_value=120.0,
        value=65.0
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

# ───────── Column 2 ─────────
with col2:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

# ───────── Column 3 ─────────
with col3:
    senior_citizen = st.checkbox("Senior Citizen")

    partner = st.checkbox("Has Partner")

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if st.button("🔮 Predict Churn Risk", type="primary"):

    # Convert booleans
    senior_citizen = int(senior_citizen)
    partner = int(partner)

    # Input dictionary
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Contract": contract,
        "InternetService": internet_service,
        "TechSupport": tech_support,
        "PaymentMethod": payment_method,
        "OnlineSecurity": online_security
    }

    df = pd.DataFrame([input_data])

    # ─────────────────────────
    # Feature Engineering
    # ─────────────────────────
    total_charges = tenure * monthly_charges

    df["TotalCharges"] = total_charges
    df["avg_charges_per_month"] = total_charges / (tenure + 1)
    df["charges_tenure_ratio"] = monthly_charges / (tenure + 1)

    # ─────────────────────────
    # One-Hot Encoding
    # ─────────────────────────
    df = pd.get_dummies(df)

    # Match training feature columns
    df = df.reindex(columns=feature_names, fill_value=0)

    # ─────────────────────────
    # Scaling
    # ─────────────────────────
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    df = df.astype(float)

    # ─────────────────────────
    # Prediction
    # ─────────────────────────
    probability = model.predict_proba(df)[0][1]

    # Risk category
    if probability < 0.3:
        risk = "Low Risk"
        color = "green"
    elif probability < 0.7:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # ─────────────────────────
    # Display Results
    # ─────────────────────────
    st.success("Prediction Complete ✅")

    st.metric(
        label="Churn Probability",
        value=f"{probability:.1%}"
    )

    st.progress(float(probability))

    st.markdown(
        f"### Risk Level: <span style='color:{color}'>{risk}</span>",
        unsafe_allow_html=True
    )