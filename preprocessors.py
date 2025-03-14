import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
import joblib
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, SGDRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_validate, StratifiedKFold, KFold, GridSearchCV, LearningCurveDisplay
from sklearn.metrics import PredictionErrorDisplay, root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.base import BaseEstimator, TransformerMixin

class NumSimpleImputer(BaseEstimator, TransformerMixin):

    def __init__(self, imputer = SimpleImputer(strategy='mean').set_output(transform= "pandas")):

        self.imputer = imputer

    def fit(self, X, y=None):

        # Example: update the imputer with learned attributes
        num_columns = X.select_dtypes(include=['float64'])
        self.num_X = X.loc[:, X.columns.isin(num_columns.columns)]

        self.imputer.fit(self.num_X)

        return self

    def transform(self, X):

        # Example: Apply transformation using the learned attribute

        X_imputed_transformed = self.imputer.transform(self.num_X)
        non_num_X = X.loc[:, ~X.columns.isin(self.num_X.columns)]

        X_imputed_transformed = pd.concat([X_imputed_transformed, non_num_X],
                                          axis=1)

        return X_imputed_transformed


class CatSimpleImputer(BaseEstimator, TransformerMixin):

    def __init__(self, imputer = SimpleImputer(strategy='constant',fill_value='NA').set_output(transform= "pandas")):

        self.imputer = imputer

    def fit(self, X, y=None):

        # Example: update the imputer with learned attributes
        cat_columns = X.select_dtypes(include=['object'])
        self.cat_X = X.loc[:, X.columns.isin(cat_columns.columns)]

        self.imputer.fit(self.cat_X)

        return self

    def transform(self, X):

        # Example: Apply transformation using the learned attribute

        X_imputed_transformed = self.imputer.transform(self.cat_X)
        non_cat_X = X.loc[:, ~X.columns.isin(self.cat_X.columns)]

        X_imputed_transformed = pd.concat([X_imputed_transformed, non_cat_X],
                                          axis=1)

        return X_imputed_transformed

class NumNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, normalizer = StandardScaler().set_output(transform= "pandas")):

        self.normalizer = normalizer

    def fit(self, X, y=None):

        #pick all numeric columns
        num_columns = X.select_dtypes(include=['float64'])
        self.num_X = X.loc[:, X.columns.isin(num_columns.columns)]

        #fit on all numeric columns
        self.normalizer.fit(self.num_X)

        return self

    def transform(self, X):

        #transform all numeric columns
        X_norm_transformed = self.normalizer.transform(self.num_X)
        non_num_X = X.loc[:, ~X.columns.isin(self.num_X.columns)]

        #join non-numeric columns back on the transformed numeric columns
        X_norm_transformed = pd.concat([X_norm_transformed, non_num_X],
                                          axis=1)

        return X_norm_transformed

class CatNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, normalizer = OrdinalEncoder().set_output(transform= "pandas")):

        self.normalizer = normalizer

    def fit(self, X, y=None):

        #pick all numeric columns
        cat_columns = X.select_dtypes(include=['object'])
        self.cat_X = X.loc[:, X.columns.isin(cat_columns.columns)]

        #fit on all numeric columns
        self.normalizer.fit(self.cat_X)

        return self

    def transform(self, X):

        #transform all numeric columns
        X_norm_transformed = self.normalizer.transform(self.cat_X)
        non_cat_X = X.loc[:, ~X.columns.isin(self.cat_X.columns)]

        #join non-numeric columns back on the transformed numeric columns
        X_norm_transformed = pd.concat([X_norm_transformed, non_cat_X],
                                          axis=1)

        return X_norm_transformed

class DtypeConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        dtype_mapping: Dictionary where keys are column names and values are the desired data types.
        """
        self.chg_to_category = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
                   'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
                   'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                   'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                   'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
                   'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                   'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
                   'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                   'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
                   'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
                'SaleType', 'SaleCondition']
        
        self.chg_to_numer = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
                'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
                'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
                'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
                'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold',
                'MoSold'
                ]

        # self.dtype_mapping = dtype_mapping

    def fit(self, X, y=None):
        # No fitting necessary; return self
        return self

    def transform(self, X):
        """
        Convert the columns in X to their respective data types as specified in dtype_mapping.
        """
        X = X.copy()  # Avoid mutating the original data
        X[self.chg_to_numer] = X[self.chg_to_numer].astype('float64')
        X[self.chg_to_category] = X[self.chg_to_category].astype('object')

        # for column, dtype in self.dtype_mapping.items():
        #     if column in X.columns:
        #         X[column] = X[column].astype(dtype)
        return X

# class feature_dtypes():

#     def __init__(self, feature_set):
#         self.feature_set = feature_set
        
#         self.chg_to_category = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape',
#                    'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
#                    'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
#                    'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
#                    'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
#                    'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
#                    'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
#                    'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
#                    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
#                    'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 
#                 'SaleType', 'SaleCondition']
        
#         self.chg_to_numer = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 
#                 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
#                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
#                 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
#                 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 
#                 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 
#                 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
#                 '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'YrSold',
#                 'MoSold'
#                 ]

#         pass

#     def enforce_dtypes(self):
#         self.feature_set[self.chg_to_numer] = self.feature_set[self.chg_to_numer].astype('float64')
#         self.feature_set[self.chg_to_category] = self.feature_set[self.chg_to_category].astype('object')

#         # self.testset[self.chg_to_numer] = self.testset[self.chg_to_numer].astype('float64')
#         # self.testset[self.chg_to_category] = self.testset[self.chg_to_category].astype('object')

#         pass


# class feat_engg():

#     def __init__(self, trainset, testset):
#         self.trainset = trainset
#         self.testset = testset
#         pass

#     def prep1(self, *cols_list):

#         numerical_selector, cat_selector = cols_list

#         preprocessors = ColumnTransformer(
#             [("scaler", StandardScaler(), numerical_selector),
#              ("enc", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_selector)]
#         ).set_output(transform='pandas')

#         return preprocessors
    
#     def prep2(self, *cols_list):

#         numerical_selector, cat_selector = cols_list

#         preprocessors = ColumnTransformer(
#             [("scaler", StandardScaler(), numerical_selector),
#              ("enc", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_selector)]
#         ).set_output(transform='pandas')

#         return preprocessors
    
#     def prep3(self, *cols_list):

#         numerical_selector, cat_selector = cols_list

#         preprocessors = ColumnTransformer(
#             [("scaler", StandardScaler(), numerical_selector),
#              ("enc", OneHotEncoder(handle_unknown='infrequent_if_exist'), cat_selector)]
#         )

#         return preprocessors
    
#     def prep4(self, *cols_list):

#         numerical_selector, cat_selector = cols_list

#         preprocessors = ColumnTransformer(
#             [("scaler", PowerTransformer(method='yeo-johnson', standardize=True), numerical_selector),
#              ("enc", OneHotEncoder(handle_unknown='infrequent_if_exist'), cat_selector)]
#         )

#         return preprocessors
    
#     def prep6(self, *cols_list):

#         def mod_ordinal_transform(X, imputer, cols_list):

#             numerical_selector, chg_to_category = cols_list

#             missing_imputer = imputer

#             testset = self.testset
#             trainset = self.trainset

#             full_data = pd.concat([trainset, testset])

#             #Fitting imputer on test+train set for consistent ordinal encoding
#             missing_imputer.fit(full_data[chg_to_category])
#             imputed_values_fulldata = pd.DataFrame(missing_imputer.transform(full_data[chg_to_category]))
#             imputed_values_fulldata.columns = missing_imputer.get_feature_names_out()

#             enc = OrdinalEncoder().fit(imputed_values_fulldata)


#             X_imputed = pd.DataFrame(missing_imputer.transform(X[chg_to_category]))
#             X_imputed.columns = missing_imputer.get_feature_names_out()

#             X_transf = pd.DataFrame(enc.transform(X_imputed))

#             X_transf.columns = enc.get_feature_names_out()
#             X_transf.index = X.index
#             X_transf = X_transf[chg_to_category]

#             return X_transf


#         numerical_selector, chg_to_category = cols_list

#         preprocessors = ColumnTransformer(
#             transformers= [
#                 ("scaler", StandardScaler(), numerical_selector),
#                 ("enc", FunctionTransformer(mod_ordinal_transform, 
#                                             validate=False, 
#                                             kw_args={'imputer': SimpleImputer(strategy='constant', fill_value='not available'),
#                                                      'cols_list': cols_list}), 
#                                     chg_to_category)
#                         ],
#             verbose_feature_names_out=False
#         ).set_output(transform='pandas')

#         return preprocessors
    
#     def prep7(self, *cols_list):

#         continous_num, time_based_num, discrete_num, chg_to_category = cols_list

#         preprocessors = ColumnTransformer(
#             transformers=[
#                 ("cont_num", PowerTransformer(method='yeo-johnson', standardize=True), continous_num),
#                 ("time_num", StandardScaler(), time_based_num),
#                 ("discrete_num", OneHotEncoder(handle_unknown='infrequent_if_exist'), discrete_num),
#                 ("enc", OneHotEncoder(handle_unknown='infrequent_if_exist'), chg_to_category)
#             ],
#         verbose_feature_names_out = False
#     )

#         return preprocessors
    

#     def prep8(self, *cols_list):

#         def mod_ordinal_transform(X, imputer, cols_list):

#             numerical_selector, chg_to_category = cols_list

#             missing_imputer = imputer

#             testset = self.testset
#             trainset = self.trainset

#             full_data = pd.concat([trainset, testset])

#             #Fitting imputer on test+train set for consistent ordinal encoding
#             missing_imputer.fit(full_data[chg_to_category])
#             imputed_values_fulldata = pd.DataFrame(missing_imputer.transform(full_data[chg_to_category]))
#             imputed_values_fulldata.columns = missing_imputer.get_feature_names_out()

#             enc = OrdinalEncoder().fit(imputed_values_fulldata)


#             X_imputed = pd.DataFrame(missing_imputer.transform(X[chg_to_category]))
#             X_imputed.columns = missing_imputer.get_feature_names_out()

#             X_transf = pd.DataFrame(enc.transform(X_imputed))

#             X_transf.columns = enc.get_feature_names_out()
#             X_transf.index = X.index
#             X_transf = X_transf[chg_to_category]

#             return X_transf


#         numerical_selector, chg_to_category = cols_list

#         preprocessors = ColumnTransformer(
#             transformers= [
#                 ("enc", FunctionTransformer(mod_ordinal_transform, 
#                                     validate=False, 
#                                     kw_args={'imputer': SimpleImputer(strategy='constant', fill_value='not available'),
#                                              'cols_list': cols_list}), 
#                 chg_to_category)
#                     ],
#                 remainder='passthrough',
#                 verbose_feature_names_out=False
#             ).set_output(transform='pandas')

#         return preprocessors
    
#     # Create new function prep9(self, *cols_list) here (within feat_engg()). This inputs a packed list of column selectors and outputs columntransformer object. This function is very similar to the other functions within feat_engg() in it's structure. AI?

# class missing_imputers():

#     def __init__(self):
#         pass

#     def imputer_1(self, *cols_list):

#         numerical_selector, cat_selector = cols_list

#         missing_imputer = ColumnTransformer(
#             [("num_mean", SimpleImputer(strategy='mean'), numerical_selector),
#              ("cat_ordinal", SimpleImputer(strategy='most_frequent'), cat_selector)]
#         ).set_output(transform='pandas')

#         return missing_imputer
    
#     def imputer_2(self, *cols_list):

#         chg_to_numer, chg_to_category = cols_list

#         missing_imputer = ColumnTransformer(
#             [("num_mean", SimpleImputer(strategy = 'constant', fill_value=0), chg_to_numer),
#              ("cat_ordinal", SimpleImputer(strategy='constant', fill_value='NA'), chg_to_category)]
#         ).set_output(transform='pandas')

#         return missing_imputer
    
#     def imputer_3(self, *cols_list):

#         chg_to_numer, chg_to_category = cols_list

#         missing_imputer = ColumnTransformer(
#             [("num_mean", KNNImputer(n_neighbors=5), chg_to_numer),
#              ("cat_ordinal", SimpleImputer(strategy='constant', fill_value='not available'), chg_to_category)],
#              verbose_feature_names_out = False
#         ).set_output(transform='pandas')

#         return missing_imputer
    
#     def imputer_4(self, *cols_list):

#         chg_to_numer, chg_to_category = cols_list

#         missing_imputer = ColumnTransformer(
#             [("num_mean", KNNImputer(n_neighbors=5), chg_to_numer),
#              ("cat_ordinal", SimpleImputer(strategy='most_frequent'), chg_to_category)],
#              verbose_feature_names_out=False
#         ).set_output(transform='pandas')

#         return missing_imputer
    
#     def imputer_5(self, *cols_list):

#         chg_to_numer, chg_to_category = cols_list

#         missing_imputer = ColumnTransformer(
#             [("num_mean", SimpleImputer(strategy='mean'), chg_to_numer),
#              ("cat_ordinal", SimpleImputer(strategy='most_frequent'), chg_to_category)],
#             verbose_feature_names_out=False
#         ).set_output(transform='pandas')

#         return missing_imputer



