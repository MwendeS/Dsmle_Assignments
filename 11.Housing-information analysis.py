# %%
# Housing Information Analysis (Ames, Iowa)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
# 1. Load the dataset from your directory
data_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Housing information data\train.csv"
data = pd.read_csv(data_path)
print("Dataset loaded. Shape:", data.shape)
# 2. Inspect basic information
print("\nINFO:")
data.info()
print("\nSalePrice Summary:")
print(data['SalePrice'].describe())
print("\nNumeric Features Summary:")
print(data.describe())
# 3. Visualize and handle missing values
print("\nVisualizing missing data…")
msno.matrix(data)
missing_ratio = data.isnull().sum() / len(data)
print("\nFeatures with missing values (>0):")
print(missing_ratio[missing_ratio > 0])
# Drop features with 5 or more missing values
data = data.dropna(axis=1, thresh=len(data) - 5)
# Drop any rows with remaining missing values
data = data.dropna()
print("\nAfter dropping, new shape:", data.shape)
# 4. Analyze target variable distribution
sns.histplot(data['SalePrice'], kde=True)
plt.title("SalePrice Distribution")
plt.show()
print(f"Kurtosis: {data['SalePrice'].kurtosis()}, Skewness: {data['SalePrice'].skew()}")
log_price = np.log(data['SalePrice'])
sns.histplot(log_price, kde=True)
plt.title("Log-transformed SalePrice Distribution")
plt.show()

print(f"Kurtosis (log): {log_price.kurtosis()}, Skewness (log): {log_price.skew()}")

# 5. Correlation analysis
corr = data.corr(numeric_only=True)
plt.figure(figsize=(10, 10))
sns.heatmap(corr, cmap='coolwarm', square=True, cbar=False)
plt.title("Correlation Matrix")
plt.show()

top10 = corr.nlargest(10, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(10, 10))
sns.heatmap(data[top10].corr(), annot=True, cmap='coolwarm', square=True)
plt.title("Top 10 Features Correlation Heatmap")
plt.show()

print("\nTop 10 features most correlated with SalePrice:")
print(top10.tolist())
# 6. Interpretation of top features
print("\nInterpretation of important features correlated with SalePrice:")
print("- OverallQual: Rates the overall material and finish of the house")
print("- GrLivArea: Above ground living area in square feet")
print("- GarageCars: Number of cars that can fit in the garage")
print("- TotalBsmtSF: Total square feet of basement area")
print("- 1stFlrSF: Square feet of the first floor")
print("These features have strong correlation with SalePrice.")
# 7. Summary Output
print("\n--- Summary ---")
print("• Checked the dataset structure and missing values")
print("• Handled missing values (removed high-missing features and rows)")
print("• Analyzed SalePrice distribution and applied log transformation")
print("• Identified top features most correlated with SalePrice")
# %%