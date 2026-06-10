import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Loan Approval System", layout="wide")

# -------------------------------
# TRAIN SYSTEM (CACHED)
# -------------------------------
@st.cache_resource
def train_system():
    df = pd.read_csv("dataset.csv")
    df.columns = [col.lower() for col in df.columns]

    if 'loan_id' in df.columns:
        df = df.drop('loan_id', axis=1)

    # Handle Missing
    for col in ['gender', 'married', 'dependents', 'self_employed', 'loan_amount_term', 'credit_history']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['loanamount'] = df['loanamount'].fillna(df['loanamount'].median())
    df = df.dropna(subset=['loan_status'])

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier(verbosity=-1)
    }

    metrics = {}

    for name, model in models.items():
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        metrics[name] = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred)
        }

    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in models.items()],
        voting='soft'
    )
    ensemble.fit(X_scaled, y)

    return ensemble, scaler, X, models, X_scaled, metrics


# -------------------------------
# INITIALIZE
# -------------------------------
ensemble, scaler, X_features, models, X_train_full, metrics = train_system()

# -------------------------------
# UI
# -------------------------------
st.title("🏦 Smart Loan Approval Dashboard")

tab1, tab2 , tab3= st.tabs(["Loan Prediction", "Correlation of Features", "Evaluation Metrics"])

# ===============================
# 📊 TAB 1: LOAN PREDICTION
# ===============================
with tab1:

    st.sidebar.header("Applicant Details")

    with st.sidebar:
        raw_app_inc = st.number_input("Monthly Income (₹)", min_value=10000, value=50000)
        raw_co_inc = st.number_input("Co-applicant Income (₹)", min_value=0, value=0)
        raw_loan_amt = st.number_input("Loan Amount (₹)", min_value=100000, value=100000)

        st.divider()

        gender = st.selectbox("Gender", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        married = st.selectbox("Married", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        dependents = st.slider("Dependents", 0, 3, 0)
        education = st.selectbox("Education", [0, 1], format_func=lambda x: "Graduate" if x==0 else "Not Grad")
        self_emp = st.selectbox("Self Employed", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        term = st.number_input("Loan Term (Months)", value=360)
        credit = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good" if x==1 else "Poor")
        area = st.radio("Property Area", [0, 1, 2], format_func=lambda x: ["Rural", "Semiurban", "Urban"][x])

        run_btn = st.button("Predict Eligibility")

    if run_btn:

        # Existing logic
        scaled_loan = raw_loan_amt / 1000 if raw_loan_amt > 1000 else raw_loan_amt

        user_input = {
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_emp,
            'applicantincome': raw_app_inc,
            'coapplicantincome': raw_co_inc,
            'loanamount': scaled_loan,
            'loan_amount_term': term,
            'credit_history': credit,
            'property_area': area
        }

        input_df = pd.DataFrame([user_input])[X_features.columns]
        input_scaled = scaler.transform(input_df)

        col1, col2 = st.columns([1, 1.5])

        # --- Predictions ---
        with col1:
            st.subheader("Model Predictions")

            for name, model in models.items():
                prob = model.predict_proba(input_scaled)[0][1]
                status = "✅ APPROVED" if prob >= 0.6 else "❌ REJECTED"
                st.write(f"{name}: {status} ({prob:.2%})")

            st.divider()

            final_prob = ensemble.predict_proba(input_scaled)[0][1]

            if final_prob >= 0.60:
                st.success(f"FINAL DECISION: APPROVED ({final_prob:.2%})")
            else:
                st.error(f"FINAL DECISION: REJECTED ({final_prob:.2%})")

        # --- SHAP ---
        with col2:
            st.subheader("AI Reasoning (SHAP)")

            explainer = shap.KernelExplainer(
                ensemble.predict_proba,
                shap.sample(X_train_full, 50)
            )

            shap_vals = explainer.shap_values(input_scaled, nsamples=100)

            vals = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0, :, 1]

            fig, ax = plt.subplots()
            feat_imp = pd.DataFrame({
                'Feature': X_features.columns,
                'Impact': vals
            }).sort_values(by='Impact')

            colors = ['red' if x < 0 else 'green' for x in feat_imp['Impact']]

            ax.barh(feat_imp['Feature'], feat_imp['Impact'], color=colors)
            ax.set_title("Feature Impact")

            st.pyplot(fig)

# ===============================
# 📈 TAB 2: Correlation
# ===============================
with tab2:

    st.subheader("Feature Correlation Heatmap")

    # Compute correlation
    corr = X_features.corr()

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

    st.pyplot(fig)
    st.divider()

    st.subheader("🔥 Feature Importance (Random Forest)")

    model = models["Random Forest"]

    importance = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": X_features.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"])
    ax.invert_yaxis()

    st.pyplot(fig)

# ===============================
# 📈 TAB 3: METRICS
# ===============================
with tab3:

    st.subheader("Model Evaluation Metrics")

    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.format("{:.4f}"))

    st.divider()

    st.subheader("Performance Comparison")

    fig, ax = plt.subplots(figsize = (5,3))
    metrics_df.plot(kind='bar', ax=ax)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")

    st.pyplot(fig)

    