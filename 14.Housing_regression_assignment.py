# %%
"""
Housing Price Regression Assignment
Goal:
- Learn basic supervised learning with regression
- Compare different regression models using scikit-learn
- Visualize predictions and evaluate model performance
Dataset:
- Ames, Iowa housing price dataset (train.csv from Kaggle)
- Features for practice: GrLivArea (ground living area), YearBuilt (construction year)
"""
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pathlib import Path

# 2. Load dataset
data_path = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Housing regression\train.csv"
data = pd.read_csv(data_path)

# 3. Preprocess dataset
# Select only two features for simplicity
features = ['GrLivArea', 'YearBuilt']
X = data[features]
y = data['SalePrice']

# Handle missing values (fill with median)
X = X.fillna(X.median())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create models
models = {
    'Linear Regression': LinearRegression(),
    'SVM': SVR(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor()
}

# 5. Train, predict, evaluate, and save results
output_dir = Path(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\14.Housing regression outputs")
output_dir.mkdir(parents=True, exist_ok=True)

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results.append({'Model': name, 'MSE': mse})
    
    # Save scatter plot
    plt.figure(figsize=(6,4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual SalePrice")
    plt.ylabel("Predicted SalePrice")
    plt.title(f"{name} Predictions")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal
    plt.savefig(output_dir / f"{name}_scatter.png")
    plt.close()
# 6. Save summary table
results_df = pd.DataFrame(results)
results_df.to_csv(output_dir / "model_comparison.csv", index=False)
# 7. Print summary
print("Model Comparison (MSE):")
print(results_df)

# %%
