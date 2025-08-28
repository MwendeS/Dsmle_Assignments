# %%
"""
Iris Data Analysis Assignment:
What this script does:
1) Loads Fisher's Iris dataset from scikit-learn
2) Builds Pandas DataFrames X (features), y (target), and combined df
3) Prints dataset overview (head, info, missing values, describe, value counts)
4) Demonstrates data extraction with .loc/.iloc
5) Creates all required visualizations using matplotlib only:
   - Pie chart of species distribution
   - Boxplots by species for each feature
   - Violin plots by species for each feature
   - 6 scatter plots for all 2-feature combinations, color-coded by species
   - Scatter-matrix alternative using pandas.plotting.scatter_matrix
   - Correlation matrix + heatmap
6) Saves figures into specified output folder
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix

def to_snake_case(names):
    """Convert scikit-learn iris.feature_names to snake_case without units."""
    mapping = {
        'sepal length (cm)': 'sepal_length',
        'sepal width (cm)': 'sepal_width',
        'petal length (cm)': 'petal_length',
        'petal width (cm)': 'petal_width',
    }
    return [mapping.get(n, n) for n in names]

def load_data():
    """Load iris dataset, return X, y, df (combined)."""
    iris = load_iris()
    feature_names = to_snake_case(iris.feature_names)

    X = pd.DataFrame(iris.data, columns=feature_names)
    y = pd.DataFrame(iris.target, columns=['Species'])
    df = pd.concat([X, y], axis=1)

    species_map = {i: name for i, name in enumerate(iris.target_names)}
    df['SpeciesName'] = df['Species'].map(species_map)

    return iris, X, y, df

def overview(df):
    print("\n=== First 4 samples ===")
    print(df.head(4))

    print("\n=== DataFrame info ===")
    df.info()

    print("\n=== Missing values per column ===")
    print(df.isnull().sum())

    print("\n=== Statistics summary (features only) ===")
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    print(df[feature_cols].describe())

    print("\n=== Count of each target (Species) ===")
    print(df['Species'].value_counts())

def extraction_examples(df):
    print("\n=== Extraction Examples ===")
    print("\nsepal_width via label (df['sepal_width']):")
    print(df['sepal_width'].head())
    print("\nsepal_width via position (df.iloc[:, 1]):")
    print(df.iloc[:, 1].head())
    print("\nRows 50–99 (df.iloc[50:100]):")
    print(df.iloc[50:100].head())
    print("\npetal_length, rows 50–99 (df.iloc[50:100, 2]):")
    print(df.iloc[50:100, 2].head())
    print("\nRows where petal_width == 0.2:")
    print(df[df['petal_width'] == 0.2].head())

def plot_pie_species(df, savedir):
    counts = df['Species'].value_counts().sort_index()
    labels = ['setosa', 'versicolor', 'virginica']

    plt.figure()
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.title('Distribution of Iris Species')
    plt.savefig(os.path.join(savedir, 'pie_species_distribution.png'), bbox_inches='tight')
    plt.close()

def plot_boxplots_by_species(df, savedir):
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    species_names = ['setosa', 'versicolor', 'virginica']

    for col in feature_cols:
        plt.figure()
        data_by_class = [df[df['SpeciesName'] == s][col].values for s in species_names]
        plt.boxplot(data_by_class, labels=species_names, showfliers=True)
        plt.title(f'Boxplot of {col} by Species')
        plt.xlabel('Species')
        plt.ylabel(col)
        plt.savefig(os.path.join(savedir, f'box_{col}_by_species.png'), bbox_inches='tight')
        plt.close()

def plot_violins_by_species(df, savedir):
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    species_names = ['setosa', 'versicolor', 'virginica']

    for col in feature_cols:
        plt.figure()
        data_by_class = [df[df['SpeciesName'] == s][col].values for s in species_names]
        plt.violinplot(dataset=data_by_class, showmeans=True, showextrema=True, showmedians=True)
        plt.xticks(np.arange(1, len(species_names) + 1), species_names)
        plt.title(f'Violin plot of {col} by Species')
        plt.xlabel('Species')
        plt.ylabel(col)
        plt.savefig(os.path.join(savedir, f'violin_{col}_by_species.png'), bbox_inches='tight')
        plt.close()

def plot_pairwise_scatter(df, savedir):
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    species_names = ['setosa', 'versicolor', 'virginica']
    combos = list(itertools.combinations(feature_cols, 2))

    for x_col, y_col in combos:
        plt.figure()
        for s in species_names:
            sub = df[df['SpeciesName'] == s]
            plt.scatter(sub[x_col], sub[y_col], label=s, alpha=0.8)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter: {y_col} vs {x_col}')
        plt.legend()
        plt.savefig(os.path.join(savedir, f'scatter_{y_col}_vs_{x_col}.png'), bbox_inches='tight')
        plt.close()

def plot_scatter_matrix(df, savedir):
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    colors = df['Species'].map({0: 0, 1: 1, 2: 2})
    axarr = scatter_matrix(df[feature_cols], figsize=(10, 10), diagonal='hist', c=colors)
    for ax in np.array(axarr).ravel():
        ax.set_xlabel(ax.get_xlabel(), rotation=45, ha='right')
        ax.set_ylabel(ax.get_ylabel(), rotation=0, ha='right')
    plt.suptitle('Scatter Matrix of Iris Features', y=1.02)
    plt.savefig(os.path.join(savedir, 'scatter_matrix.png'), bbox_inches='tight')
    plt.close()

def plot_corr_and_heatmap(df, savedir):
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    corr = df[feature_cols].corr()
    corr.to_csv(os.path.join(savedir, 'correlation_matrix.csv'), index=True)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr.values, interpolation='nearest')
    plt.title('Correlation Heatmap')
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha='right')
    plt.yticks(range(len(feature_cols)), feature_cols)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'correlation_heatmap.png'), bbox_inches='tight')
    plt.close()

def main():
    iris, X, Y, df = load_data()
    savedir = r"C:\Users\admin\DevWorkspace\DSMLE_Course\Assignments\Outputs\10.Iris data output"
    os.makedirs(savedir, exist_ok=True)

    overview(df)
    extraction_examples(df)

    plot_pie_species(df, savedir)
    plot_boxplots_by_species(df, savedir)
    plot_violins_by_species(df, savedir)
    plot_pairwise_scatter(df, savedir)
    plot_scatter_matrix(df, savedir)
    plot_corr_and_heatmap(df, savedir)

    print(f"\nAll figures and outputs have been saved in: {savedir}\n")

if __name__ == '__main__':
    main()
# %%
