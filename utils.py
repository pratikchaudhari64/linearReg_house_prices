import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.metrics import PredictionErrorDisplay

def residual_plots(model, data, y):

    """
    Plots residuals and related graphs for a set of model, and dataset (features, target). can you

    Parameters:
    model (pd.Series): scikit-learn model
    data (dataframe): dataframe of all features
    y (pd.Series): The target variable for model to test residuals against
    """

    predicted_y = model.predict(data)
    # y.iloc[:, 0].to_list()
    residuals = y.iloc[:, 0].to_list() - predicted_y
    residuals = pd.DataFrame(residuals, columns=['residuals'])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Visualizations')

    # 1. Histogram of predicted vs actual
    axes[0, 0].hist(predicted_y, bins=50, color='dodgerblue', alpha=0.5, label='Predicted')
    axes[0, 0].hist(y, bins=50, color='orange', alpha=0.5, label='Actual')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].set_title('Predicted vs Actual Histogram')

    # 2. Histogram of residuals
    residuals.plot.hist(bins=50, ax=axes[0, 1], color='purple', alpha=0.7)
    axes[0, 1].set_title('Residuals Histogram')
    axes[0, 1].set_ylabel('Frequency')

    # 3. Residuals vs Predicted
    PredictionErrorDisplay.from_estimator(
        model,
        data,
        y.squeeze(),
        kind='residual_vs_predicted',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Residuals vs Predicted')

    # 4. Actual vs Predicted
    PredictionErrorDisplay.from_estimator(
        model,
        data,
        y.squeeze(),
        kind='actual_vs_predicted',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Actual vs Predicted')

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    pass


def plot_outlier_analysis(column, target):
    """
    Plots a histogram of the given column with overlaid Z-scores, quantiles, IQR, 
    IQR-based outlier bounds, and mean, and shows the correlation with the target variable.

    Parameters:
    column (pd.Series): The data column to analyze.
    target (pd.Series): The target variable for correlation analysis.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Calculate Z-scores
    z_scores = (column - column.mean()) / column.std()

    # Calculate quantiles
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1

    # Calculate bounds for outliers using IQR method
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Calculate mean
    mean_value = column.mean()

    # Calculate correlation with the target
    correlation = column.corr(target)

    # Plot histogram with Z-scores, quantiles, IQR, outlier bounds, and mean on the same graph
    plt.figure(figsize=(12, 6))

    # Histogram of the original data
    sns.histplot(column, bins=30, kde=True, color='lightblue', edgecolor='black')

    # Add vertical lines for quantiles
    plt.axvline(Q1, color='r', linestyle='--', linewidth=2, label=f'Q1 (25th percentile) = {Q1:.2f}')
    plt.axvline(Q3, color='r', linestyle='--', linewidth=2, label=f'Q3 (75th percentile) = {Q3:.2f}')
    plt.axvline(column.median(), color='g', linestyle='--', linewidth=2, label=f'Median (Q2) = {column.median():.2f}')

    # Add horizontal lines for IQR
    plt.axhline(IQR, color='purple', linestyle='--', linewidth=2, label=f'IQR = {IQR:.2f}')

    # Add vertical lines for Z-scores
    mean = column.mean()
    std = column.std()
    plt.axvline(mean + 3 * std, color='orange', linestyle='--', linewidth=2, label=f'Z = 3 ({mean + 3 * std:.2f})')
    plt.axvline(mean - 3 * std, color='orange', linestyle='--', linewidth=2, label=f'Z = -3 ({mean - 3 * std:.2f})')

    # Add vertical lines for IQR-based outlier bounds
    plt.axvline(lower_bound, color='blue', linestyle='--', linewidth=2, label=f'Q1 - 1.5*IQR = {lower_bound:.2f}')
    plt.axvline(upper_bound, color='blue', linestyle='--', linewidth=2, label=f'Q3 + 1.5*IQR = {upper_bound:.2f}')

    # Add vertical line for mean
    plt.axvline(mean_value, color='magenta', linestyle='-', linewidth=2, label=f'Mean = {mean_value:.2f}')

    # Add correlation text
    plt.text(0.95, 0.95, f'Correlation: {correlation:.2f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f" {column.name}: Outlier Analysis")
    plt.legend()

    plt.show()
