# %%
# Credit Risk Analysis Assignment
# Goal:
# - Learn analytical techniques using credit information data
# - Acquire fundamentals of data analysis through Kaggle
# Dataset: Home Credit Default Risk (from Kaggle)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from pathlib import Path

# 1. Load Data
data_path = Path(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Credit analysis data\application_train.csv")
df = pd.read_csv(data_path)

# 2. Basic EDA
def basic_eda(df, out_dir):
    eda_lines = []
    eda_lines.append("=== HEAD (first 5 rows) ===")
    eda_lines.append(df.head(5).to_string())

    eda_lines.append("\n=== INFO ===")
    buf = io.StringIO()
    df.info(buf=buf)
    eda_lines.append(buf.getvalue())

    eda_lines.append("\n=== DESCRIBE (numeric) ===")
    eda_lines.append(df.describe().T.to_string())

    with open(out_dir / "eda_summary.txt", "w", encoding="utf8") as f:
        f.write("\n\n".join(eda_lines))

    print(f"[INFO] EDA summary written to {out_dir / 'eda_summary.txt'}")

# 3. Handle Missing Values
def handle_missing(df, threshold=0.4):
    missing_ratio = df.isnull().mean()
    to_drop = missing_ratio[missing_ratio > threshold].index
    df_clean = df.drop(columns=to_drop)
    print(f"[INFO] Dropped {len(to_drop)} columns with >{threshold*100}% missing values")
    return df_clean

# 4. Correlation Analysis
def correlation_analysis(df, out_dir):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (numeric features)")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png")
    plt.close()
    print(f"[INFO] Correlation heatmap saved to {out_dir / 'correlation_heatmap.png'}")

# 5. Target Variable Distribution
def target_distribution(df, out_dir):
    plt.figure(figsize=(6,4))
    sns.countplot(x="TARGET", data=df)
    plt.title("Target Distribution (0 = Non-Default, 1 = Default)")
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png")
    plt.close()
    print(f"[INFO] Target distribution plot saved to {out_dir / 'target_distribution.png'}")

# ==== Main Execution ====
def main():
    out_dir = Path(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\12.Credit risk outputs")
    out_dir.mkdir(parents=True, exist_ok=True)  # Ensure all parent folders are created

    # Basic EDA
    basic_eda(df, out_dir)

    # Missing values handling
    df_clean = handle_missing(df)

    # Correlation analysis
    correlation_analysis(df_clean, out_dir)

    # Target distribution
    target_distribution(df_clean, out_dir)

    print(f"[INFO] Analysis complete. Check the folder: {out_dir}")

if __name__ == "__main__":
    main()
# %%
