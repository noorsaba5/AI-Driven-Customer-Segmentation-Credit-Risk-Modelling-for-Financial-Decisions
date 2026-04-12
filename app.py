import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="AI-Driven Customer Segmentation & Credit Risk Modelling",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Driven Customer Segmentation & Credit Risk Modelling")
st.markdown("An interactive Streamlit app for **customer segmentation** and **credit risk prediction**.")

# -----------------------------
# File paths
# -----------------------------
DATA_PATH = "data"
CC_FILE = os.path.join(DATA_PATH, "CC GENERAL.csv")
LOAN_FILE = os.path.join(DATA_PATH, "loan_portfolio.csv")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    cc_df = pd.read_csv(CC_FILE)
    loan_df = pd.read_csv(LOAN_FILE)
    return cc_df, loan_df

cc_df, loan_df = load_data()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    ["Project Overview", "Customer Segmentation", "Credit Risk Prediction"]
)

# -----------------------------
# Project Overview
# -----------------------------
if section == "Project Overview":
    st.subheader("Project Overview")
    st.write(
        """
        This app demonstrates two machine learning applications in finance:

        1. **Customer Segmentation** using KMeans clustering on credit card behavioural data  
        2. **Credit Risk Prediction** using a Random Forest classifier on loan portfolio data

        It supports financial decision-making by helping identify:
        - distinct customer groups
        - potentially high-risk loans
        - key drivers of default
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Credit Card Records", f"{cc_df.shape[0]:,}")
        st.metric("Credit Card Features", cc_df.shape[1])
    with col2:
        st.metric("Loan Records", f"{loan_df.shape[0]:,}")
        st.metric("Loan Features", loan_df.shape[1])

    st.markdown("---")
    st.write("### Sample Data")
    st.write("**Credit Card Dataset**")
    st.dataframe(cc_df.head())

    st.write("**Loan Portfolio Dataset**")
    st.dataframe(loan_df.head())

# -----------------------------
# Customer Segmentation
# -----------------------------
elif section == "Customer Segmentation":
    st.subheader("Customer Segmentation")

    st.write(
        """
        In this section, customers are grouped into behavioural segments using **KMeans clustering**.
        """
    )

    df_cluster = cc_df.copy()
    if "CUST_ID" in df_cluster.columns:
        df_cluster = df_cluster.drop("CUST_ID", axis=1)

    df_cluster = df_cluster.fillna(df_cluster.median(numeric_only=True))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_cluster)

    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=8, value=4)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    df_cluster["Cluster"] = clusters

    st.write("### Cluster Summary")
    cluster_summary = df_cluster.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(cluster_summary)

    st.write("### Customer Segmentation Visualisation")
    x_axis = st.selectbox("Select X-axis", [col for col in df_cluster.columns if col != "Cluster"], index=2)
    y_axis = st.selectbox("Select Y-axis", [col for col in df_cluster.columns if col != "Cluster"], index=12)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(df_cluster[x_axis], df_cluster[y_axis], c=df_cluster["Cluster"])
    ax.set_title("Customer Segmentation")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    st.pyplot(fig)

    st.write("### Segment Insights")
    st.markdown(
        """
        Typical segment interpretation may include:
        - **High-value customers** with high purchases and strong credit limits
        - **Low-activity customers** with minimal transactions
        - **Cash-advance-heavy customers** who may show higher financial stress
        - **Moderate users** with balanced financial behaviour
        """
    )

# -----------------------------
# Credit Risk Prediction
# -----------------------------
elif section == "Credit Risk Prediction":
    st.subheader("Credit Risk Prediction")

    st.write(
        """
        In this section, a **Random Forest model** is used to predict whether a loan is likely to default.
        """
    )

    df_risk = loan_df.copy()

    drop_cols = ["loan_id", "origination_date", "maturity_date", "default_date"]
    df_risk = df_risk.drop(columns=drop_cols, errors="ignore")

    if "defaulted" not in df_risk.columns:
        st.error("The 'defaulted' column was not found in the loan dataset.")
        st.stop()

    df_risk = pd.get_dummies(df_risk, drop_first=True)

    X = df_risk.drop("defaulted", axis=1)
    y = df_risk["defaulted"]

    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X, y)

    st.write("### Enter Loan Details")

    # Safer defaults
    def get_default(col, fallback=0.0):
        return float(X[col].median()) if col in X.columns else fallback

    maturity_months = st.number_input("Maturity Months", min_value=1, value=int(get_default("maturity_months", 12)))
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=int(get_default("credit_score", 650)))
    ead = st.number_input("EAD", min_value=0.0, value=get_default("ead", 100000.0))
    coupon_rate = st.number_input("Coupon Rate", min_value=0.0, value=get_default("coupon_rate", 0.05), format="%.4f")
    leverage = st.number_input("Leverage", min_value=0.0, value=get_default("leverage", 2.0))
    interest_coverage = st.number_input("Interest Coverage", min_value=0.0, value=get_default("interest_coverage", 3.0))
    debt_to_equity = st.number_input("Debt to Equity", min_value=0.0, value=get_default("debt_to_equity", 1.5))
    pd_annual = st.number_input("PD Annual", min_value=0.0, max_value=1.0, value=min(max(get_default("pd_annual", 0.1), 0.0), 1.0), format="%.4f")
    lgd = st.number_input("LGD", min_value=0.0, max_value=1.0, value=min(max(get_default("lgd", 0.4), 0.0), 1.0), format="%.4f")
    el = st.number_input("EL", min_value=0.0, value=get_default("el", 10000.0))
    unexpected_loss = st.number_input("Unexpected Loss", min_value=0.0, value=get_default("unexpected_loss", 5000.0))
    rwa = st.number_input("RWA", min_value=0.0, value=get_default("rwa", 50000.0))
    survival_months = st.number_input("Survival Months", min_value=0.0, value=get_default("survival_months", 24.0))
    recovery_rate = st.number_input("Recovery Rate", min_value=0.0, max_value=1.0, value=min(max(get_default("recovery_rate", 0.5), 0.0), 1.0), format="%.4f")
    loss_given_default = st.number_input(
        "Loss Given Default",
        min_value=0.0,
        max_value=1.0,
        value=min(max(get_default("loss_given_default", 0.5), 0.0), 1.0),
        format="%.4f"
    )

    # Build base input row
    input_data = pd.DataFrame([{
        "maturity_months": maturity_months,
        "credit_score": credit_score,
        "ead": ead,
        "coupon_rate": coupon_rate,
        "leverage": leverage,
        "interest_coverage": interest_coverage,
        "debt_to_equity": debt_to_equity,
        "pd_annual": pd_annual,
        "lgd": lgd,
        "el": el,
        "unexpected_loss": unexpected_loss,
        "rwa": rwa,
        "survival_months": survival_months,
        "recovery_rate": recovery_rate,
        "loss_given_default": loss_given_default
    }])

    # Align columns to trained model
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict Risk"):
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        expected_loss = pd_annual * lgd * ead

        st.write("### Prediction Result")
        if prediction == 1:
            st.error(f"High Risk of Default")
        else:
            st.success(f"Low Risk of Default")

        st.write(f"**Probability of Default:** {probability:.2%}")
        st.write(f"**Expected Loss:** {expected_loss:,.2f}")

        st.write("### Model Insight")
        st.markdown(
            """
            This prediction is based on key financial features such as:
            - credit score
            - leverage
            - debt-to-equity ratio
            - probability of default
            - loss given default
            """
        )