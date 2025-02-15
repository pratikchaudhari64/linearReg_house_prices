# AI: refactor this code, such that i can compare scores of 2 models separately. So, run DecionTreeRegressor on the dataset, then also run LinearRegresson on the dataset....
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(X_train, X_test, y_train, y_test):
    # Create and train the model with Decision Tree Regressor
    model1 = DecisionTreeRegressor(max_depth=5, min_samples_split=10, random_state=42)
    model1.fit(X_train, y_train)

    # Make predictions for Decision Tree Regressor
    y_pred1 = model1.predict(X_test)

    # Create and train the model with Linear Regression
    model2 = LinearRegression()
    model2.fit(X_train, y_train)

    # Make predictions for Linear Regression
    y_pred2 = model2.predict(X_test)

    # Evaluate both models
    mse1 = mean_squared_error(y_test, y_pred1)
    r2_1 = r2_score(y_test, y_pred1)
    mse2 = mean_squared_error(y_test, y_pred2)
    r2_2 = r2_score(y_test, y_pred2)

    print(f"Test MSE (Decision Tree Regressor): {mse1}")
    print(f"R² score (Decision Tree Regressor): {r2_1}")
    print(f"Test MSE (Linear Regression): {mse2}")
    print(f"R² score (Linear Regression): {r2_2}")

# Fetch and split the dataset
data = load_boston()
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.Series(data['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

evaluate_models(X_train, X_test, y_train, y_test)
