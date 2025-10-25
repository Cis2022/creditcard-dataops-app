import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import logging, time, os

# Setup logging
logging.basicConfig(filename="data_pipeline.log",
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

os.makedirs("plots/univariate", exist_ok=True)
os.makedirs("plots/bivariate", exist_ok=True)

def run_pipeline(df):
    start = time.time()
    logging.info("=== Pipeline started ===")

    result = {}
    result["summary"] = df.describe(include="all")
    result["dtypes"] = df.dtypes.to_dict()
    result["missing"] = df.isnull().sum().to_dict()

    # Handle missing numeric values
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    logging.info("Missing numeric values imputed")

    # Normalize numeric columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        logging.info("Numeric columns normalized")

    # --- Univariate Analysis ---
    for col in num_cols:
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"plots/univariate/{col}_hist.png")
        plt.close()

        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"plots/univariate/{col}_box.png")
        plt.close()

    # --- Bivariate Analysis ---
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("plots/bivariate/correlation_heatmap.png")
    plt.close()

    if "Class" in df.columns:
        sns.countplot(x="Class", data=df, palette="Set2")
        plt.title("Fraud (1) vs Non-Fraud (0)")
        plt.savefig("plots/bivariate/class_distribution.png")
        plt.close()

        for col in num_cols[:5]:
            sns.boxplot(x=df["Class"], y=df[col])
            plt.title(f"{col} vs Class")
            plt.savefig(f"plots/bivariate/{col}_vs_Class.png")
            plt.close()

        X = df.drop("Class", axis=1).select_dtypes(include=[np.number])
        y = df["Class"]
        if len(y.unique()) > 1:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X, y)
            imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            imp.head(10).plot(kind="barh")
            plt.title("Top 10 Feature Importances")
            plt.gca().invert_yaxis()
            plt.savefig("plots/bivariate/feature_importance.png")
            plt.close()
            result["feature_importance"] = imp.head(10).to_dict()

    end = time.time()
    logging.info(f"Pipeline completed in {end - start:.2f}s")

    with open("data_pipeline.log") as f:
        result["logs"] = f.read()[-3000:]

    return result
