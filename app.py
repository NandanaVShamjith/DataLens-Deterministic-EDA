"""
DataLens – Exploratory Data Analysis App

This application performs end-to-end EDA including:
- Data cleaning and validation
- Dataset health scoring
- Deterministic insight generation
- Optional natural-language explanations for users

All analytics are rule-based and reproducible.
Language models are used only to explain results, not to compute them.
"""



# Core libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import csv
import tempfile

# Data preprocessing
from sklearn.preprocessing import OrdinalEncoder

# App utilities
from analysis_utils import generate_summary_text
from llm_utils import configure_llm, generate_insights
from report_utils import generate_pdf_report
import tempfile

# Ui
from PIL import Image


def compute_deterministic_insights(df):
    insights = {}
    # Identify columns with high missing values
    missing_pct = df.isnull().mean() * 100

    high_missing = (
        missing_pct[missing_pct > 10]
            .sort_values(ascending=False)
    )
    # Detect categorical columns with too many unique values
    insights["high_missing_columns"] = high_missing

    insights["high_cardinality_columns"] = {
        col: df[col].nunique()
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() > 50
    }
    # Analyze skewness in numeric features
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 0:
        insights["skewed_features"] = (
            numeric_df.skew()
                .abs()
                .sort_values(ascending=False)
                .head(5)
        )
    else:
        insights["skewed_features"] = pd.Series(dtype=float)

    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr().abs()
        insights["strong_correlations"] = (
            corr.where((corr > 0.7) & (corr < 1))
                .stack()
                .sort_values(ascending=False)
                .head(10)
        )
    else:
        insights["strong_correlations"] = pd.Series(dtype=float)

    return insights

# Controlled language model call (used only for explanations)

def safe_llm_call(prompt):
    try:
        configure_llm()
        response = generate_insights(prompt)
        return response, None
    except Exception as e:
        return None, str(e)


st.set_page_config(page_title="DataLens", layout="wide")
summary_text = None
# Load logo
logo = Image.open(r"C:\Users\nandana\Downloads\DataLens1.png")

# Create 3 equal columns
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(logo, use_container_width=False, width=350)  # adjust width as needed

st.write("Upload a CSV, xls, xlsx file to explore, clean, visualize, and engineer features dynamically.")

uploaded_file = st.file_uploader(
    "Upload a data file",
    type=["csv", "xls", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    else:
        df = pd.read_excel(uploaded_file)


    # ORIGINAL DATA PREVIEW

    st.subheader(" Original Data Preview")
    st.dataframe(df.head())


    # DATASET SHAPE

    st.subheader(" Dataset Shape")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # ======================
    # DATA TYPES
    # ======================
    st.subheader(" Column Data Types")
    dtypes_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str)
    })
    st.dataframe(dtypes_df)


    # MISSING VALUES

    st.subheader(" Missing Values Analysis (Before Cleaning)")
    missing_df = pd.DataFrame({
        "Column Name": df.columns,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_df)


    # Clean and standardize the dataset

    st.subheader(" Data Cleaning")
    cleaned_df = df.copy()

    # Remove duplicates
    cleaned_df.drop_duplicates(inplace=True)

    # Fill missing values
    for col in cleaned_df.columns:
        if cleaned_df[col].isnull().sum() > 0:
            if cleaned_df[col].dtype == "object":
                cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
            else:
                cleaned_df[col].fillna(cleaned_df[col].median(), inplace=True)

    # Cap outliers using IQR
    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        Q1 = cleaned_df[col].quantile(0.25)
        Q3 = cleaned_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        cleaned_df[col] = cleaned_df[col].clip(lower, upper)

    st.success("Cleaning completed: duplicates removed, missing values filled, outliers capped.")


    # CLEANED DATA PREVIEW

    st.subheader(" Cleaned Data Preview")
    st.dataframe(cleaned_df.head())

    # DATASET HEALTH METRICS (DETERMINISTIC)

    missing_pct = cleaned_df.isnull().sum().sum() / cleaned_df.size * 100
    duplicate_pct = (len(df) - len(cleaned_df)) / len(df) * 100
    numeric_ratio = 0  # default
    if uploaded_file is not None:
        numeric_ratio = len(cleaned_df.select_dtypes(include=["number"]).columns) / cleaned_df.shape[1]

    health_score = round(
        100
        - (missing_pct * 2)
        - (duplicate_pct * 1.5)
        + (numeric_ratio * 10),
        2
    )
    health_score = max(min(health_score, 100), 0)

    deterministic_insights = compute_deterministic_insights(cleaned_df)

    st.subheader(" DataLens Insights ")

    high_missing = deterministic_insights["high_missing_columns"]

    if high_missing.empty:
        st.success("No columns exceed missing value threshold.")
    else:
        st.warning("Columns with high missing values")
        st.dataframe(high_missing)

    if deterministic_insights["high_cardinality_columns"]:
        st.warning("⚠️ High-cardinality categorical columns")
        st.json(deterministic_insights["high_cardinality_columns"])
    else:
        st.success("✅ No high-cardinality categorical columns")

    if not deterministic_insights["skewed_features"].empty:
        st.warning("⚠️ Skewed numeric features")
        st.dataframe(deterministic_insights["skewed_features"].round(2))
    else:
        st.success("✅ No heavily skewed numeric features")

    if not deterministic_insights["strong_correlations"].empty:
        st.warning("⚠️ Strong correlations detected")
        st.dataframe(deterministic_insights["strong_correlations"].round(2))

    # SUMMARY STATISTICS

    st.subheader("Summary Statistics ")
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found.")
    else:
        st.dataframe(cleaned_df[numeric_cols].describe().T)


    # Explore column distributions

    st.subheader("Column Visualizations")
    selected_col = st.selectbox("Select a column to visualize", cleaned_df.columns)

    if cleaned_df[selected_col].dtype == "object" or cleaned_df[selected_col].nunique() < 20:
        st.bar_chart(cleaned_df[selected_col].value_counts())
    else:
        st.bar_chart(cleaned_df[selected_col].value_counts(bins=10))


    # Compare outliers before and after cleaning

    st.subheader("Outlier Visualization (Before vs After Cleaning)")
    for col in numeric_cols:
        st.write(f"Column: {col}")

        fig1, ax1 = plt.subplots(figsize=(6, 1))  # width=6, height=1
        ax1.boxplot(df[col], vert=False)
        ax1.set_title(f"Original {col}", fontsize=10)
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(6, 1))  # smaller height
        ax2.boxplot(cleaned_df[col], vert=False)
        ax2.set_title(f"Cleaned {col}", fontsize=10)
        st.pyplot(fig2)


    # CORRELATION ANALYSIS

    st.subheader(" Correlation Analysis")
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns for correlation.")
    else:
        corr_matrix = cleaned_df[numeric_cols].corr()
        st.dataframe(corr_matrix)
        st.bar_chart(corr_matrix.abs().mean().sort_values(ascending=False))


    # DATETIME FEATURE DETECTION

    st.subheader(" Datetime Feature Extraction")

    datetime_cols = []
    for col in cleaned_df.select_dtypes(include=["object"]).columns:
        try:
            parsed = pd.to_datetime(cleaned_df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                datetime_cols.append(col)


        except:
            pass

    if len(datetime_cols) == 0:
        st.info("No datetime columns detected.")
    else:
        for col in datetime_cols:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col])
            cleaned_df[f"{col}_year"] = cleaned_df[col].dt.year
            cleaned_df[f"{col}_month"] = cleaned_df[col].dt.month
            cleaned_df[f"{col}_day"] = cleaned_df[col].dt.day
            cleaned_df[f"{col}_weekday"] = cleaned_df[col].dt.weekday
            st.success(f"Datetime features extracted from {col}")


    # FEATURE ENGINEERING

    st.subheader("Feature Engineering ")

    categorical_cols = cleaned_df.select_dtypes(include=["object"]).columns.tolist()

    encoding_type = st.selectbox(
        "Select encoding method",
        ["None", "One-Hot Encoding", "Ordinal Encoding"]
    )

    selected_cols = st.multiselect(
        "Select categorical columns to encode",
        categorical_cols
    )

    if encoding_type == "One-Hot Encoding" and selected_cols:
        cleaned_df = pd.get_dummies(cleaned_df, columns=selected_cols)
        st.success("One-hot encoding applied.")

    if encoding_type == "Ordinal Encoding" and selected_cols:
        oe = OrdinalEncoder()
        cleaned_df[selected_cols] = oe.fit_transform(cleaned_df[selected_cols])
        st.success("Ordinal encoding applied.")
    st.subheader(" Dataset Health Score")

    st.metric("Dataset Health Score", f"{health_score} / 100")

    if health_score > 80:
        st.success("Excellent dataset quality ")
    elif health_score > 60:
        st.warning("Moderate quality – some improvements recommended")
    else:
        st.error("Poor data quality – significant cleaning needed")

    numeric_cols = cleaned_df.select_dtypes(include=["number"]).columns


    # FINAL DATA PREVIEW
    st.subheader(" Final Dataset Preview")
    st.dataframe(cleaned_df.head())


# Column Insights
    st.subheader(" Column Insights")

    selected_column = st.selectbox(
        "Select a column for AI explanation",
        cleaned_df.columns
    )

    if st.button("Explain Selected Column"):
        column_summary = f"""
        Column Name: {selected_column}
        Data Type: {cleaned_df[selected_column].dtype}
        Missing Values: {cleaned_df[selected_column].isnull().sum()}
        Unique Values: {cleaned_df[selected_column].nunique()}
        Sample Values: {cleaned_df[selected_column].dropna().unique()[:10]}
        """

        try:
            configure_llm()
            explanation = generate_insights(
                f"Explain this dataset column for business analysis:\n{column_summary}"
            )
            st.info(explanation)
        except:
            st.warning("LLM quota exceeded. Please try again later.")


    # CREATE SUMMARY TEXT FOR LLM

    summary_text = generate_summary_text(cleaned_df)


# Insight explanation

if uploaded_file is not None:
    st.subheader("Dataset Insights")

    if st.button("Explain Insights"):
        prompt = f"""
        You are a senior data analyst.

        TASK:
        Explain the following PRE-COMPUTED insights in clear business language.

        RULES:
        - Do NOT add new insights
        - Do NOT assume missing information
        - Only explain what is given

        Insights:
        {deterministic_insights}
        """

        with st.spinner("DataLens is explaining insights..."):
            explanation, error = safe_llm_call(prompt)

        if error:
            st.warning(error)
        else:
            st.info(explanation)

            st.download_button(
                " Download AI Insight Report (TXT)",
                explanation,
                file_name="ai_eda_insights.txt",
                mime="text/plain"
            )

            # PDF DOWNLOAD (ONLY IF SUCCESS)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                generate_pdf_report(explanation, tmp_file.name)

                with open(tmp_file.name, "rb") as f:
                    st.download_button(
                        " Download Insight Report (PDF)",
                        f,
                        file_name="ai_eda_insights.pdf",
                        mime="application/pdf"
                    )





# Machine learning readiness check

# ML readiness risks
ml_risks = []

if uploaded_file is not None:
    # numeric_ratio is already defined
    if numeric_ratio < 0.3:
        ml_risks.append("Low numeric feature ratio")

    if cleaned_df.shape[0] < 500:
        ml_risks.append("Small dataset size")

    if missing_pct > 5:
        ml_risks.append("High missing value percentage")

    if st.button("Explain ML Readiness"):
        ml_prompt = f"""
        Explain the ML readiness assessment below.
        Do NOT add new risks.

        Health Score: {health_score}
        Identified Risks: {ml_risks}
        """

        with st.spinner("DataLens is analyzing ML readiness..."):
            response, error = safe_llm_call(ml_prompt)

        if error:
            st.warning(error)
        else:
            st.info(response)




# # Ask questions about the dataset (explanation-only)

if uploaded_file is not None:
    st.subheader("Ask DataLens")
    user_question = st.text_input(
        "Ask a business or analysis question",
        placeholder="e.g. What are the biggest risks?"
    )

    if st.button("Ask Question"):
        if summary_text is None:
            st.warning("Please upload and process a dataset first.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            ask_prompt = f"""
            You are a senior data analyst.

            RULES:
            - Answer in maximum 6 bullet points
            - Use ONLY the dataset summary
            - Do NOT fabricate or assume
            - If unknown, clearly say so
            STRICT RULE:
            If answer cannot be derived from provided summary,
            respond: "This information is not available from the data."


            Dataset Summary:
            {summary_text}

            User Question:
            {user_question}
            """

            with st.spinner("Analyzing...."):
                answer, error = safe_llm_call(ask_prompt)

            if error:
                st.warning(error)
            else:
                st.markdown("### DataLens Answer")
                st.info(answer)


# DOWNLOAD CLEANED DATA
if uploaded_file is not None:
    st.subheader("Download Cleaned Data")

    # CSV download
    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )



