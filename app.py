import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="AI-Driven Customer Segmentation & Credit Risk Modelling",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI-Driven Customer Segmentation & Credit Risk Modelling")
st.caption("Customer segmentation and credit risk prediction for financial decision-making.")

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CC_FILE = DATA_DIR / "CC GENERAL.csv"
LOAN_FILE = DATA_DIR / "loan_portfolio.csv"


# --------------------------------------------------
# Data loading
# --------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load required datasets safely."""
    missing_files = [str(p.name) for p in [CC_FILE, LOAN_FILE] if not p.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required file(s): {', '.join(missing_files)}. "
            f"Make sure they are inside the data folder."
        )

    cc_df = pd.read_csv(CC_FILE)
    loan_df = pd.read_csv(LOAN_FILE)
    return cc_df, loan_df


@st.cache_resource
def train_credit_risk_model(loan_df: pd.DataFrame):
    """
    Prepare training data and fit a Random Forest model.
    Returns model, feature matrix columns, and processed data.
    """
    df = loan_df.copy()

    drop_cols = ["loan_id", "origination_date", "maturity_date", "default_date"]
    df = df.drop(columns=drop_cols, errors="ignore")

    if "defaulted" not in df.columns:
        raise KeyError("The loan dataset must contain a 'defaulted' column.")

    df_model = pd.get_dummies(df, drop_first=True)

    X = df_model.drop("defaulted", axis=1)
    y = df_model["defaulted"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    return model, X.columns.tolist(), X


@st.cache_resource
def build_cluster_model(cc_df: pd.DataFrame, n_clusters: int):
    """
    Fit a clustering pipeline for the customer segmentation dataset.
    Returns processed dataframe, scaler, fitted KMeans model, and scaled data.
    """
    df = cc_df.copy()

    if "CUST_ID" in df.columns:
        df = df.drop("CUST_ID", axis=1)

    df = df.fillna(df.median(numeric_only=True))

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    return df, scaler, kmeans, scaled_data


# --------------------------------------------------
# Safe load
# --------------------------------------------------
try:
    cc_df, loan_df = load_data()
except Exception as e:
    st.error(f"❌ {e}")
    st.stop()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Overview", "Customer Segmentation", "Credit Risk Prediction"],
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if section == "Overview":
    st.subheader("Project Overview")

    st.write(
        """
        This application demonstrates two machine learning use cases in finance:

        **1. Customer Segmentation**
        - Groups customers using behavioural credit card data
        - Helps identify high-value, low-activity, and potentially risky segments

        **2. Credit Risk Prediction**
        - Predicts whether a loan is likely to default
        - Supports lending and financial risk decisions
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Customer Records", f"{len(cc_df):,}")
        st.metric("Customer Features", cc_df.shape[1])

    with col2:
        st.metric("Loan Records", f"{len(loan_df):,}")
        st.metric("Loan Features", loan_df.shape[1])

    st.markdown("---")
    st.write("### Dataset Preview")

    tab1, tab2 = st.tabs(["Customer Data", "Loan Data"])
    with tab1:
        st.dataframe(cc_df.head(), use_container_width=True)
    with tab2:
        st.dataframe(loan_df.head(), use_container_width=True)

# --------------------------------------------------
# Customer Segmentation
# --------------------------------------------------
elif section == "Customer Segmentation":
    st.subheader("Customer Segmentation")

    st.write(
        """
        This section uses **KMeans clustering** to group customers based on spending,
        credit usage, payment behaviour, and cash advance activity.
        """
    )

    n_clusters = st.slider("Select number of clusters", min_value=2, max_value=8, value=4)

    try:
        df_cluster, _, _, _ = build_cluster_model(cc_df, n_clusters)
    except Exception as e:
        st.error(f"Unable to build clustering model: {e}")
        st.stop()

    st.write("### Cluster Summary")
    cluster_summary = df_cluster.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(cluster_summary, use_container_width=True)

    st.write("### Cluster Visualisation")

    plot_columns = [c for c in df_cluster.columns if c != "Cluster"]
    default_x = plot_columns.index("PURCHASES") if "PURCHASES" in plot_columns else 0
    default_y = plot_columns.index("CREDIT_LIMIT") if "CREDIT_LIMIT" in plot_columns else min(1, len(plot_columns) - 1)

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", plot_columns, index=default_x)
    with col2:
        y_axis = st.selectbox("Y-axis", plot_columns, index=default_y)

    fig, ax = plt.subplots(figsize=(9, 5))
    scatter = ax.scatter(
        df_cluster[x_axis],
        df_cluster[y_axis],
        c=df_cluster["Cluster"],
    )
    ax.set_title("Customer Segmentation")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    st.pyplot(fig)

    st.write("### Segment Interpretation")
    st.markdown(
        """
        Typical cluster patterns may include:

        - **High-value customers** with strong purchases and high credit limits
        - **Low-activity customers** with limited engagement
        - **Cash-advance-heavy customers** who may indicate financial stress
        - **Moderate users** with balanced behaviour
        """
    )

# --------------------------------------------------
# Credit Risk Prediction
# --------------------------------------------------
elif section == "Credit Risk Prediction":
    st.subheader("Credit Risk Prediction")

    st.write(
        """
        This section uses a **Random Forest classifier** trained on loan portfolio data
        to estimate whether a loan is likely to default.
        """
    )

    try:
        model, feature_columns, X_reference = train_credit_risk_model(loan_df)
    except Exception as e:
        st.error(f"Unable to train credit risk model: {e}")
        st.stop()

    st.write("### Enter Loan Details")

    def feature_default(name: str, fallback: float) -> float:
        """Return median feature value from training data or fallback."""
        if name in X_reference.columns:
            return float(X_reference[name].median())
        return float(fallback)

    with st.form("risk_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            maturity_months = st.number_input(
                "Maturity Months",
                min_value=1,
                value=int(feature_default("maturity_months", 24)),
            )
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=int(feature_default("credit_score", 650)),
            )
            ead = st.number_input(
                "EAD",
                min_value=0.0,
                value=feature_default("ead", 100000.0),
                format="%.2f",
            )
            coupon_rate = st.number_input(
                "Coupon Rate",
                min_value=0.0,
                value=feature_default("coupon_rate", 4.0),
                format="%.4f",
            )
            leverage = st.number_input(
                "Leverage",
                min_value=0.0,
                value=feature_default("leverage", 5.0),
                format="%.4f",
            )

        with col2:
            interest_coverage = st.number_input(
                "Interest Coverage",
                min_value=0.0,
                value=feature_default("interest_coverage", 3.0),
                format="%.4f",
            )
            debt_to_equity = st.number_input(
                "Debt to Equity",
                min_value=0.0,
                value=feature_default("debt_to_equity", 1.5),
                format="%.4f",
            )
            pd_annual = st.number_input(
                "PD Annual",
                min_value=0.0,
                max_value=1.0,
                value=min(max(feature_default("pd_annual", 0.10), 0.0), 1.0),
                format="%.4f",
            )
            lgd = st.number_input(
                "LGD",
                min_value=0.0,
                max_value=1.0,
                value=min(max(feature_default("lgd", 0.40), 0.0), 1.0),
                format="%.4f",
            )
            el = st.number_input(
                "EL",
                min_value=0.0,
                value=feature_default("el", 10000.0),
                format="%.2f",
            )

        with col3:
            unexpected_loss = st.number_input(
                "Unexpected Loss",
                min_value=0.0,
                value=feature_default("unexpected_loss", 5000.0),
                format="%.2f",
            )
            rwa = st.number_input(
                "RWA",
                min_value=0.0,
                value=feature_default("rwa", 50000.0),
                format="%.2f",
            )
            survival_months = st.number_input(
                "Survival Months",
                min_value=0,
                value=int(feature_default("survival_months", 24)),
            )
            recovery_rate = st.number_input(
                "Recovery Rate",
                min_value=0.0,
                max_value=1.0,
                value=min(max(feature_default("recovery_rate", 0.50), 0.0), 1.0),
                format="%.4f",
            )
            loss_given_default = st.number_input(
                "Loss Given Default",
                min_value=0.0,
                max_value=1.0,
                value=min(max(feature_default("loss_given_default", 0.50), 0.0), 1.0),
                format="%.4f",
            )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_data = pd.DataFrame(
            [{
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
                "loss_given_default": loss_given_default,
            }]
        )

        # Align with training columns
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        expected_loss = pd_annual * lgd * ead

        st.write("### Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("High Risk of Default")
            else:
                st.success("Low Risk of Default")

        with col2:
            st.metric("Probability of Default", f"{probability:.2%}")

        with col3:
            st.metric("Expected Loss", f"{expected_loss:,.2f}")

        st.markdown("---")
        st.write("### Interpretation")
        st.write(
            """
            This prediction is based on the financial characteristics entered above.
            In general, higher probability of default, higher LGD, and larger EAD
            contribute to greater expected financial loss.
            """
        )