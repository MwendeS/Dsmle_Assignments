# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
# Load only a subset of columns for baseline
train_data = pd.read_csv(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Credit info\application_train.csv",
                         usecols=['SK_ID_CURR','TARGET','AMT_CREDIT','AMT_ANNUITY','AMT_INCOME_TOTAL','DAYS_BIRTH'])
test_data = pd.read_csv(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Data\Credit info\application_test.csv",
                        usecols=['SK_ID_CURR','AMT_CREDIT','AMT_ANNUITY','AMT_INCOME_TOTAL','DAYS_BIRTH'])

# Fill missing values quickly
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)
# Features and target
X = train_data.drop(['SK_ID_CURR','TARGET'], axis=1)
y = train_data['TARGET']
X_test = test_data.drop(['SK_ID_CURR'], axis=1)
# Train Logistic Regression with solver optimized for speed
model = LogisticRegression(max_iter=500, solver='liblinear')
model.fit(X, y)
# Predict
preds = model.predict_proba(X_test)[:,1]
# Save submission
submission = pd.DataFrame({
    "SK_ID_CURR": test_data["SK_ID_CURR"],
    "TARGET": preds
})
submission.to_csv(r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\15.HomeCreditDefaultRisk_Submission.csv", index=False)
print("Done! Submission saved.")
# %%
