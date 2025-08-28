# %%
# Iris classification assignment
# Goal: Binary classification (virginica vs versicolor) using sepal_length & petal_length;
# also compare multiple classifiers and do a multiclass run on all features.
# Save outputs to outputs/ and print evaluation summary.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore")
RND = 42
OUTDIR = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\13.Iris classification outputs"
os.makedirs(OUTDIR, exist_ok=True)

# %% (cell 2) — helper: decision region visualization
def decision_region(X, y, model, step=0.01, title='Decision region', xlabel='X0', ylabel='X1', target_names=None, savepath=None):
    """
    X: 2D numpy array
    y: 1D array of labels (binary or two classes expected here)
    model: trained classifier with .predict
    """
    if target_names is None:
        target_names = np.unique(y).astype(str)
    scatter_color = ['red', 'blue']
    contourf_color = ['lightcoral', 'lightblue']
    n_class = len(np.unique(y))
    # meshgrid limits
    f0_min, f0_max = np.min(X[:,0]) - 0.5, np.max(X[:,0]) + 0.5
    f1_min, f1_max = np.min(X[:,1]) - 0.5, np.max(X[:,1]) + 0.5
    mesh_f0, mesh_f1 = np.meshgrid(np.arange(f0_min, f0_max, step),
                                   np.arange(f1_min, f1_max, step))
    mesh = np.c_[np.ravel(mesh_f0), np.ravel(mesh_f1)]
    y_pred = model.predict(mesh).reshape(mesh_f0.shape)

    plt.figure(figsize=(7,6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # background
    plt.contourf(mesh_f0, mesh_f1, y_pred, alpha=0.3, cmap=ListedColormap(contourf_color[:n_class]))
    plt.contour(mesh_f0, mesh_f1, y_pred, colors='k', linewidths=1, alpha=0.5)
    for i, target in enumerate(np.unique(y)):
        plt.scatter(X[y==target][:, 0], X[y==target][:, 1], s=60, color=scatter_color[i%len(scatter_color)], label=target_names[i], edgecolor='k')
    patches = [mpatches.Patch(color=scatter_color[i%len(scatter_color)], label=target_names[i]) for i in range(n_class)]
    plt.legend(handles=patches)
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.show()


# %% (cell 3) — load iris dataset and make dataframe
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = [iris.target_names[t] for t in iris.target]

# quick view
print("Dataset head:")
print(df.head())

# %% (cell 4) — TASK: select classes and features for binary classification
# We select Iris versicolor (1) and Iris virginica (2), and features sepal length & petal length.
binary_df = df[df['target'].isin([1,2])].copy()  # 1=versicolor, 2=virginica
# Map to 0/1 labels for convenience: 0=versicolor, 1=virginica
binary_df['y'] = (binary_df['target'] == 2).astype(int)
X_binary = binary_df[['sepal length (cm)', 'petal length (cm)']].values
y_binary = binary_df['y'].values

print("\nBinary dataset shape:", X_binary.shape)

# %% (cell 5) — EDA: scatter plot, boxplot, violin plot, scatter matrix
sns.set(style="whitegrid")
# scatter
plt.figure(figsize=(6,5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target_name', data=binary_df, s=80)
plt.title("Scatter: sepal_length vs petal_length (versicolor vs virginica)")
plt.savefig(os.path.join(OUTDIR, "scatter_binary.png"), dpi=150, bbox_inches='tight')
plt.show()

# boxplots
plt.figure(figsize=(8,4))
binary_df[['sepal length (cm)', 'petal length (cm)', 'target_name']].boxplot(by='target_name', layout=(1,2))
plt.suptitle('')
plt.savefig(os.path.join(OUTDIR, "boxplots_binary.png"), dpi=150, bbox_inches='tight')
plt.show()

# violin plots
plt.figure(figsize=(8,4))
sns.violinplot(x='target_name', y='petal length (cm)', data=binary_df)
plt.title("Violin: petal length by class")
plt.savefig(os.path.join(OUTDIR, "violin_petal_length.png"), dpi=150, bbox_inches='tight')
plt.show()

# scatter matrix (pairplot) for the two features
sns.pairplot(binary_df[['sepal length (cm)', 'petal length (cm)', 'target_name']], hue='target_name', height=3)
plt.savefig(os.path.join(OUTDIR, "pairplot_binary.png"), dpi=150, bbox_inches='tight')
plt.show()

# %% (cell 6) — Train/Validation split (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.25, random_state=RND, stratify=y_binary)
print("\nTrain/test shapes:", X_train.shape, X_test.shape)

# %% (cell 7) — Standardization (fit on train only)
scaler = StandardScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# For comparison, keep non-standardized versions too:
X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

# %% (cell 8) — helper: train, predict, evaluate
def evaluate_model(clf, X_tr, y_tr, X_te, y_te, labels=['versicolor','virginica']):
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    cm = confusion_matrix(y_te, y_pred)
    print(f"Model: {clf.__class__.__name__}")
    print("Accuracy: {:.3f} Precision: {:.3f} Recall: {:.3f} F1: {:.3f}".format(acc, prec, rec, f1))
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", classification_report(y_te, y_pred, target_names=labels))
    return {'model': clf, 'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'cm': cm}

# %% (cell 9) — Define the classifiers to try
classifiers = {
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'LogisticRegression': LogisticRegression(random_state=RND, solver='liblinear'),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RND),
    'DecisionTree': DecisionTreeClassifier(random_state=RND),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=RND)
}

# %% (cell 10) — TRAIN & EVALUATE: WITHOUT standardization (raw features)
results_raw = {}
print("\n--- Evaluation (WITHOUT standardization) ---")
for name, clf in classifiers.items():
    print("\n==>", name)
    res = evaluate_model(clf, X_train_raw, y_train, X_test_raw, y_test, labels=['versicolor','virginica'])
    results_raw[name] = res

# %% (cell 11) — TRAIN & EVALUATE: WITH standardization
results_std = {}
print("\n--- Evaluation (WITH standardization) ---")
for name, clf in classifiers.items():
    # For tree-based models, scaling doesn't matter much, but we still compare.
    print("\n==>", name)
    res = evaluate_model(clf, X_train_std, y_train, X_test_std, y_test, labels=['versicolor','virginica'])
    results_std[name] = res

# %% (cell 12) — Plot decision regions for each model (on standardized data)
for name, res in results_std.items():
    model = res['model']
    title = f"{name} (std) — decision region"
    savepath = os.path.join(OUTDIR, f"decision_{name.replace(' ','_').replace('(','').replace(')','')}.png")
    # decision_region expects 2D features; show regions on training data to see boundaries
    decision_region(X_train_std, y_train, model, step=0.02, title=title, xlabel='sepal length (std)', ylabel='petal length (std)', target_names=['versicolor','virginica'], savepath=savepath)

# %% (cell 13) — Compare and summarize metrics in a DataFrame
def summary_df(results_dict):
    rows = []
    for name, res in results_dict.items():
        rows.append({
            'model': name,
            'accuracy': res['acc'],
            'precision': res['prec'],
            'recall': res['rec'],
            'f1': res['f1']
        })
    return pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)

print("\nSummary WITHOUT standardization:")
print(summary_df(results_raw))

print("\nSummary WITH standardization:")
print(summary_df(results_std))
summary_std = summary_df(results_std)
summary_std.to_csv(os.path.join(OUTDIR, "summary_with_standardization.csv"), index=False)

# %% (cell 14) — Multiclass classification using all features
print("\n--- Multiclass classification (all 3 classes) using all 4 features ---")
X_multi = df[iris.feature_names].values
y_multi = df['target'].values  # 0,1,2
Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(X_multi, y_multi, test_size=0.25, random_state=RND, stratify=y_multi)

# Standardize
scaler_m = StandardScaler().fit(Xtr_m)
Xtr_m_std = scaler_m.transform(Xtr_m)
Xte_m_std = scaler_m.transform(Xte_m)

# Use RandomForest and LogisticRegression (multiclass)
clf_rf = RandomForestClassifier(n_estimators=100, random_state=RND)
clf_log = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=200, random_state=RND)

print("\nRandomForest (multiclass):")
clf_rf.fit(Xtr_m_std, ytr_m)
pred_rf = clf_rf.predict(Xte_m_std)
print(classification_report(yte_m, pred_rf, target_names=iris.target_names))
cm_rf = confusion_matrix(yte_m, pred_rf)
print("Confusion matrix:\n", cm_rf)

print("\nLogisticRegression (multiclass):")
clf_log.fit(Xtr_m_std, ytr_m)
pred_log = clf_log.predict(Xte_m_std)
print(classification_report(yte_m, pred_log, target_names=iris.target_names))
cm_log = confusion_matrix(yte_m, pred_log)
print("Confusion matrix:\n", cm_log)

# %% (cell 15) — Save some extra plots: decision tree visualization (binary) and feature importance from RF
# Train a decision tree on standardized binary training set for plotting
dt = DecisionTreeClassifier(random_state=RND).fit(X_train_std, y_train)
plt.figure(figsize=(10,6))
plot_tree(dt, feature_names=['sepal length (std)','petal length (std)'], class_names=['versicolor','virginica'], filled=True)
plt.title("Decision Tree (binary) trained on standardized features")
plt.savefig(os.path.join(OUTDIR, "decision_tree_binary.png"), dpi=150, bbox_inches='tight')
plt.show()

# Feature importances (multiclass RF)
clf_rf_full = RandomForestClassifier(n_estimators=200, random_state=RND).fit(Xtr_m_std, ytr_m)
importances = clf_rf_full.feature_importances_
feat_imp = pd.Series(importances, index=iris.feature_names).sort_values(ascending=False)
print("\nFeature importances (RandomForest, multiclass):")
print(feat_imp)
plt.figure(figsize=(6,4))
feat_imp.plot(kind='bar')
plt.ylabel("Importance")
plt.title("Feature importances (multiclass RF)")
plt.savefig(os.path.join(OUTDIR, "feature_importances_rf.png"), dpi=150, bbox_inches='tight')
plt.show()

# %% (cell 16) — End: Save key CSV summaries
summary_std.to_csv(os.path.join(OUTDIR, "model_summary_standardized.csv"), index=False)
print(f"\nAll key outputs saved to folder: {OUTDIR}")

# %%
