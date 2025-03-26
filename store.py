from preprocessors import NumSimpleImputer, CatSimpleImputer, NumNormalizer, CatNormalizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression

def create_pipeline_v1():
    """
    First version of preprocessing pipeline.
    Example: Basic imputers and scaling.
    """
    pipeline = Pipeline([
        ('num_imputer', NumSimpleImputer(imputer=SimpleImputer(strategy='mean').set_output(transform="pandas"))),
        ('cat_imputer', CatSimpleImputer(imputer=SimpleImputer(strategy='constant', fill_value='NA').set_output(transform="pandas"))),
        ('scaler', NumNormalizer(normalizer=StandardScaler().set_output(transform="pandas"))),
        ('encoder', CatNormalizer(normalizer=OrdinalEncoder().set_output(transform="pandas")))
    ])
    return pipeline

def create_pipeline_v2():
    """
    Second version of preprocessing pipeline.
    Example: Adding extra transformations (e.g., feature scaling or encoding).
    """
    pipeline = Pipeline([
        ('num_imputer', NumSimpleImputer(imputer=SimpleImputer(strategy='median').set_output(transform="pandas"))),
        ('cat_imputer', CatSimpleImputer(imputer=SimpleImputer(strategy='constant', fill_value='Missing').set_output(transform="pandas"))),
        ('scaler', NumNormalizer(normalizer=StandardScaler().set_output(transform="pandas"))),
        ('encoder', CatNormalizer(normalizer=OrdinalEncoder().set_output(transform="pandas")))
    ])
    return pipeline

# Add more pipelines as needed for experimentation.


# -------------------------------
# MODEL DEFINITIONS
# -------------------------------
def create_linear_regression_model():
    """
    Create and return a Linear Regression model.
    """
    return LinearRegression()

def create_ridge_model(alpha=1.0):
    """
    Create and return a Ridge Regression model.
    """
    return Ridge(alpha=alpha)

def create_lasso_model(alpha=0.1):
    """
    Create and return a Lasso Regression model.
    """
    return Lasso(alpha=alpha)

# -------------------------------
# STORING MULTIPLE OPTIONS
# -------------------------------
# Organize pipelines and models in dictionaries for centralized access
pipelines = {
    'pipeline_v1': create_pipeline_v1,
    'pipeline_v2': create_pipeline_v2
}

models = {
    'linear_regression': create_linear_regression_model,
    'ridge_regression': lambda alpha=1.0: create_ridge_model(alpha=alpha),  # Lambda for flexibility
    'lasso_regression': lambda alpha=0.1: create_lasso_model(alpha=alpha)
}
