import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    shapiro,
    levene,
    ttest_ind,
    mannwhitneyu,
    f_oneway,
)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Page config
st.set_page_config(
    page_title="Medical Insurance Cost Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — this is what makes it look less "default Streamlit"
st.markdown("""
<style>
    /* Overall font */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }

    /* Header banner */
    .main-header {
        background: linear-gradient(120deg, #0f4c81 0%, #1f77b4 50%, #38a3d1 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 6px 18px rgba(15, 76, 129, 0.25);
    }
    .main-header h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.02rem;
        margin: 0;
    }

    /* KPI metric cards */
    div[data-testid="stMetric"] {
        border: 1px solid #eef1f5;
        border-left: 5px solid #1f77b4;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #4a5568;
    }

    /* Section subheaders */
    h2, h3 {
        color: #0f4c81;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
    }

    /* Result banners */
    .result-box {
        padding: 1rem 1.3rem;
        border-radius: 12px;
        font-size: 1.05rem;
        margin-top: 0.6rem;
    }
    .result-reject {
        background-color: #fdecea;
        border-left: 5px solid #e63946;
        color: #7a1f1f;
    }
    .result-fail {
        background-color: #eaf7ec;
        border-left: 5px solid #2a9d8f;
        color: #1c5c53;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(120deg, #1f77b4, #38a3d1);
        padding: 1.6rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 6px 18px rgba(31, 119, 180, 0.3);
    }
    .prediction-card h2 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
    }
    .prediction-card p {
        color: rgba(255,255,255,0.9);
        margin: 0.2rem 0 0 0;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-size: 1.02rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = {"yes": "#e63946", "no": "#1f77b4"}

# Header
st.markdown("""
<div class="main-header">
    <h1> Medical Insurance Cost Analysis Dashboard</h1>
    <p>Exploratory Data Analysis · Hypothesis Testing · Multiple Linear Regression · Diagnostics</p>
</div>
""", unsafe_allow_html=True)

# Data loading (cached)
@st.cache_data
def load_data():
    return pd.read_csv("data/insurance.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Could not find `insurance.csv`. Make sure it's in the same folder as this app.")
    st.stop()

# Regression Model (cached so it isn't refit on every widget interaction)
@st.cache_resource
def fit_model(data: pd.DataFrame):
    data_encoded = pd.get_dummies(data, drop_first=True, dtype=int)
    y = data_encoded["charges"]
    X = data_encoded.drop("charges", axis=1)
    X = sm.add_constant(X)
    fitted = sm.OLS(y, X).fit()
    return fitted, X, y

model, X, y = fit_model(df)

tab1, tab2, tab3 = st.tabs([
    " Data Exploration",
    " Hypothesis Testing",
    " Prediction & Diagnostics",
])

# TAB 1 — DATA EXPLORATION
with tab1:
    st.header("Data Exploration")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{df.shape[0]:,}")
    k2.metric("Avg. Charges", f"${df['charges'].mean():,.0f}")
    k3.metric("Avg. Age", f"{df['age'].mean():.1f} yrs")
    k4.metric("Avg. BMI", f"{df['bmi'].mean():.1f}")

    st.write("")

    with st.expander(" View Raw Dataset", expanded=False):
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.write(f"**Rows:** {df.shape[0]}")
        c2.write(f"**Columns:** {df.shape[1]}")

    with st.expander(" Summary Statistics", expanded=False):
        st.dataframe(df.describe().style.background_gradient(cmap="Blues"), use_container_width=True)

    st.divider()

    # ---- Sidebar filters ----
    st.sidebar.header("🔧 Filters")
    st.sidebar.caption("Adjust these to update every chart below.")

    age_range = st.sidebar.slider(
        "Age",
        int(df.age.min()),
        int(df.age.max()),
        (int(df.age.min()), int(df.age.max())),
    )

    smoker_filter = st.sidebar.multiselect(
        "Smoker",
        df["smoker"].unique(),
        default=df["smoker"].unique(),
    )

    region_filter = st.sidebar.multiselect(
        "Region",
        df["region"].unique(),
        default=df["region"].unique(),
    )

    st.sidebar.divider()
    if st.sidebar.button("↺ Reset Filters", use_container_width=True):
        st.rerun()

    filtered_df = df[
        (df["age"] >= age_range[0])
        & (df["age"] <= age_range[1])
        & (df["smoker"].isin(smoker_filter))
        & (df["region"].isin(region_filter))
    ]

    st.subheader(f"Filtered Dataset  ·  {filtered_df.shape[0]:,} records")
    st.dataframe(filtered_df, use_container_width=True, height=220)

    numerical_cols = filtered_df.select_dtypes(include=["int64", "float64"]).columns

    if filtered_df.empty:
        st.warning("No records match the current filters — try widening your selection.")
    else:
        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader(" Distribution Plot")
            selected_col = st.selectbox("Select Numerical Variable", numerical_cols)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.histplot(filtered_df[selected_col], kde=True, ax=ax, color="#1f77b4")
            ax.set_title(f"Distribution of {selected_col}", fontsize=13, fontweight="bold")
            ax.set_xlabel(selected_col)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col_b:
            st.subheader(" Box Plot")
            box_col = st.selectbox("Select Variable", numerical_cols, key="box")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.boxplot(x=filtered_df[box_col], ax=ax, color="#38a3d1")
            ax.set_title(f"Box Plot of {box_col}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader(" Scatter Plot")
            x_axis = st.selectbox("Select X-axis", numerical_cols, key="x")
            y_axis = st.selectbox("Select Y-axis", numerical_cols, index=min(1, len(numerical_cols) - 1), key="y")
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.scatterplot(
                data=filtered_df, x=x_axis, y=y_axis, hue="smoker",
                palette=PALETTE, ax=ax, alpha=0.75, edgecolor="white",
            )
            ax.set_title(f"{x_axis} vs {y_axis}", fontsize=13, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with col_d:
            st.subheader(" Correlation Heatmap")
            corr = filtered_df[numerical_cols].corr()
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax, fmt=".2f")
            ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

# TAB 2 — HYPOTHESIS TESTING
with tab2:
    st.header(" Hypothesis Testing Lab")

    test = st.selectbox(
        "Select Hypothesis Test",
        ("Smokers vs Non-Smokers", "Charges Across Regions (ANOVA)"),
    )

    st.divider()

    if test == "Smokers vs Non-Smokers":
        st.subheader("Independent Samples Comparison")
        c1, c2 = st.columns(2)
        c1.info("**H₀:** Mean medical charges are the same for smokers and non-smokers.")
        c2.warning("**H₁:** Mean medical charges are different for smokers and non-smokers.")

        smoker_charges = df[df["smoker"] == "yes"]["charges"]
        non_smoker_charges = df[df["smoker"] == "no"]["charges"]

        smoker_p = shapiro(smoker_charges).pvalue
        non_smoker_p = shapiro(non_smoker_charges).pvalue
        levene_p = levene(smoker_charges, non_smoker_charges).pvalue
        alpha = 0.05

        if smoker_p > alpha and non_smoker_p > alpha:
            equal_var = levene_p > alpha
            statistic, p_value = ttest_ind(smoker_charges, non_smoker_charges, equal_var=equal_var)
            test_used = "Independent Two-Sample t-test"
        else:
            statistic, p_value = mannwhitneyu(smoker_charges, non_smoker_charges, alternative="two-sided")
            test_used = "Mann-Whitney U Test"

        st.subheader("Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test Used", test_used)
        m2.metric("Test Statistic", f"{statistic:.4f}")
        m3.metric("P-value", f"{p_value:.6f}")

        # Visual comparison
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="smoker", y="charges", palette=PALETTE, ax=ax)
        ax.set_title("Charges by Smoking Status", fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

        if p_value < alpha:
            st.markdown(
                '<div class="result-box result-reject"> <b>Reject H₀:</b> '
                'There is a significant difference in medical charges between smokers and non-smokers.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box result-fail">ℹ <b>Fail to Reject H₀:</b> '
                'No significant difference was found.</div>',
                unsafe_allow_html=True,
            )

    elif test == "Charges Across Regions (ANOVA)":
        st.subheader("One-Way ANOVA")
        c1, c2 = st.columns(2)
        c1.info("**H₀:** Mean medical charges are equal across all regions.")
        c2.warning("**H₁:** At least one region has a different mean medical charge.")

        northwest = df[df["region"] == "northwest"]["charges"]
        northeast = df[df["region"] == "northeast"]["charges"]
        southwest = df[df["region"] == "southwest"]["charges"]
        southeast = df[df["region"] == "southeast"]["charges"]

        f_stat, p_value = f_oneway(northwest, northeast, southwest, southeast)

        st.subheader("Results")
        m1, m2 = st.columns(2)
        m1.metric("F Statistic", f"{f_stat:.4f}")
        m2.metric("P-value", f"{p_value:.6f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x="region", y="charges", palette="Blues", ax=ax)
        ax.set_title("Charges by Region", fontsize=13, fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

        if p_value < 0.05:
            st.markdown(
                '<div class="result-box result-reject"> <b>Reject H₀:</b> '
                'Average medical charges differ significantly across regions.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box result-fail">ℹ <b>Fail to Reject H₀:</b> '
                'Average medical charges do not differ significantly across regions.</div>',
                unsafe_allow_html=True,
            )

# TAB 3 — PREDICTION & DIAGNOSTICS
with tab3:
    st.header(" Live Prediction & Diagnostics")

    st.subheader("Enter Customer Details")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 64, 30)
            sex = st.selectbox("Sex", ("female", "male"))
        with c2:
            bmi = st.slider("BMI", 15.0, 55.0, 25.0)
            smoker = st.selectbox("Smoker", ("no", "yes"))
        with c3:
            children = st.slider("Children", 0, 5, 0)
            region = st.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    sex_male = 1 if sex == "male" else 0
    smoker_yes = 1 if smoker == "yes" else 0
    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    input_df = pd.DataFrame({
        "const": [1],
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [sex_male],
        "smoker_yes": [smoker_yes],
        "region_northwest": [region_northwest],
        "region_southeast": [region_southeast],
        "region_southwest": [region_southwest],
    })[X.columns]  # ensure column order matches the fitted model

    prediction = model.predict(input_df)[0]
    prediction_result = model.get_prediction(input_df)
    prediction_summary = prediction_result.summary_frame(alpha=0.05)
    lower = prediction_summary["obs_ci_lower"].iloc[0]
    upper = prediction_summary["obs_ci_upper"].iloc[0]

    st.write("")
    p1, p2 = st.columns([1, 1])
    with p1:
        st.markdown(f"""
        <div class="prediction-card">
            <p>Predicted Medical Charges</p>
            <h2>${prediction:,.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="prediction-card" style="background: linear-gradient(120deg, #2a9d8f, #38a3d1);">
            <p>95% Prediction Interval</p>
            <h2 style="font-size:1.4rem;">${lower:,.0f} &nbsp;–&nbsp; ${upper:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---- Model quality metrics ----
    st.subheader("Model Fit")
    q1, q2, q3 = st.columns(3)
    q1.metric("R²", f"{model.rsquared:.3f}")
    q2.metric("Adj. R²", f"{model.rsquared_adj:.3f}")
    q3.metric("F-statistic p-value", f"{model.f_pvalue:.2e}")
    st.divider()
    st.subheader("Diagnostic Plots")
    predictions = model.predict(X)
    residuals = y - predictions

    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.scatterplot(x=predictions, y=residuals, ax=ax, alpha=0.6, color="#1f77b4", edgecolor="white")
        ax.axhline(y=0, color="#e63946", linestyle="--")
        ax.set_title("Residuals vs Fitted", fontsize=13, fontweight="bold")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with d2:
        fig = plt.figure(figsize=(6.5, 4.5))
        sm.qqplot(residuals, line="45", fit=True, ax=plt.gca())
        plt.title("Q-Q Plot", fontsize=13, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with st.expander(" Multicollinearity (VIF)"):
        vif_data = pd.DataFrame()
        vif_features = X.drop(columns=["const"])
        vif_data["Feature"] = vif_features.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, X.columns.get_loc(col))
            for col in vif_features.columns
        ]
        st.dataframe(
            vif_data.style.background_gradient(cmap="Reds", subset=["VIF"]),
            use_container_width=True,
        )
        st.caption("VIF > 5–10 typically signals problematic multicollinearity between predictors.")

    with st.expander("📄 Full Regression Summary"):
        st.text(model.summary())
