# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_autorefresh import st_autorefresh

from data_pipeline import load_dataframe_from_upload, run_pipeline_from_dataframe

# Auto-refresh every 2 minutes
st_autorefresh(interval=2*60*1000, key="datarefresh")

st.set_page_config(layout="wide", page_title="Credit Card Fraud EDA Pipeline")
st.title("Credit Card Fraud EDA Pipeline (Phase 1)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to see the analysis.")
    st.stop()

# Try to load df
try:
    df = load_dataframe_from_upload(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded CSV: {e}")
    st.stop()

st.subheader("Data Preview")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head())

# Basic dtypes + missing info
with st.expander("Data types & missing values"):
    st.write(df.dtypes)
    miss = df.isnull().sum().sort_values(ascending=False)
    st.write("Missing values (top):")
    st.write(miss[miss > 0].head(20))

# Attempt to run pipeline with helpful Streamlit feedback
st.subheader("Running Phase 1 Pipeline (Validation & Model Training)")
try:
    result = run_pipeline_from_dataframe(df)
except Exception as e:
    st.error("Pipeline failed during validation/training.")
    st.exception(e)
    st.stop()

# Display training results
st.success("Pipeline completed successfully!")
st.write("Target column detected:", result.get('target_col'))
st.write("Class distribution (raw counts):", result.get('class_counts'))
st.write("Model accuracy on test set:", result['metrics']['accuracy'])

# Show classification report in readable form
st.subheader("Classification Report")
cr = result['metrics']['classification_report']
st.json(cr)

# ----------------------------
# EDA plots
# ----------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Univariate (show only first 10 to avoid long runs)
st.subheader("Univariate Distributions (numeric features)")
cols_to_plot = numeric_cols[:10]
for col in cols_to_plot:
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), bins=30, kde=True, ax=ax)
    ax.set_title(f"{col} distribution")
    st.pyplot(fig)

# Correlation heatmap (subset to numeric columns)
st.subheader("Correlation Heatmap (numeric columns)")
if len(numeric_cols) >= 2:
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.info("Not enough numeric columns to show correlation heatmap.")

# Top features vs target (if 'Class' exists)
target_col = result.get('target_col')
if target_col in df.columns and target_col in numeric_cols:
    # If class is numeric we can show scatter for top correlations
    st.subheader("Top features vs Target (by correlation)")
    # compute correlation of each numeric feature with target (only numeric features)
    corr_with_target = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_features = corr_with_target.index[1:6] if len(corr_with_target) > 1 else corr_with_target.index[:5]
    for feat in top_features:
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[feat], y=df[target_col], ax=ax, alpha=0.6)
        ax.set_title(f"{feat} vs {target_col}")
        st.pyplot(fig)
else:
    st.info("Target column is non-numeric or not present in numeric features; skipping feature-target scatterplots.")

# Normalized plots (show first 6 numeric features)
st.subheader("Normalized distributions (first 6 numeric cols)")
from sklearn.preprocessing import MinMaxScaler
if numeric_cols:
    cols_norm = numeric_cols[:6]
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[cols_norm].fillna(0)), columns=cols_norm)
    for col in cols_norm:
        fig, ax = plt.subplots()
        sns.histplot(scaled[col], bins=30, kde=True, ax=ax)
        ax.set_title(f"Normalized {col}")
        st.pyplot(fig)
else:
    st.info("No numeric columns to normalize.")

# Fraud vs Non-Fraud chart (if target exists)
st.subheader("Fraud vs Non-Fraud Distribution")
if target_col in df.columns:
    try:
        fig, ax = plt.subplots()
        sns.countplot(x=target_col, data=df, ax=ax)
        ax.set_title('Fraud vs Non-Fraud Count')
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Unable to draw countplot for target column: {e}")
else:
    st.info("Target column not detected; skipping Fraud vs Non-Fraud chart.")
