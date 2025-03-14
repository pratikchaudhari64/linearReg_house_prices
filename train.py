# get training folds csv.
# describe prep and model pipeline
# write cross validation to pick one fold as validation set and rest as training set
# for each iteration of cross validation, apply pipeline (prep steps + model) on training set and calculate error metrics on validation set

import pandas as pd
import preprocessors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression


#Convert dtypes as a custom transformer
def train_pipeline_and_model(X, y):

    # feat_dtypes= preprocessors.feature_dtypes(feature_set=X)
    # feat_dtypes.enforce_dtypes()

    prep_pipeline = Pipeline([
        ('dtype_converter', preprocessors.DtypeConverter()),
        ('num_imputer', preprocessors.NumSimpleImputer(imputer=SimpleImputer(strategy='mean').set_output(transform="pandas"))),  # numeric columns imputer
        ('cat_imputer', preprocessors.CatSimpleImputer(imputer= SimpleImputer(strategy='constant',fill_value='NA').set_output(transform="pandas"))),  # categorical imputer
        ('standardscaler', preprocessors.NumNormalizer(normalizer= StandardScaler().set_output(transform="pandas"))),
        ('catencoder', preprocessors.CatNormalizer(normalizer = OrdinalEncoder().set_output(transform= "pandas"))) 
        ])

    X_transformed = prep_pipeline.fit_transform(X)

    model = LinearRegression().fit(X_transformed, y)

    print("...model training finished.")

    return prep_pipeline, model

if __name__ == "__main__":

    fold_training_data = pd.read_csv("./inputs/train_folds.csv")
    fold_training_data.set_index('Id', inplace=True)
    testset = pd.read_csv("./inputs/test.csv")

    y = fold_training_data[['SalePrice']]
    X = fold_training_data.drop(['SalePrice', 'kfold'], axis = 1)
    
    folds = fold_training_data['kfold'].unique()

    feat_dtypes= preprocessors.feature_dtypes(trainset=fold_training_data, testset=testset)
    feat_dtypes.enforce_dtypes()

    # create preprocessing steps here

    # prep_pipeline = Pipeline([
    #     ('num_imputer', preprocessors.NumSimpleImputer(imputer=SimpleImputer(strategy='mean').set_output(transform="pandas"))),  # numeric columns imputer
    #     ('cat_imputer', preprocessors.CatSimpleImputer(imputer= SimpleImputer(strategy='constant',fill_value='NA').set_output(transform="pandas"))),  # categorical imputer
    #     ('standardscaler', preprocessors.NumNormalizer(normalizer= StandardScaler().set_output(transform="pandas"))),
    #     ('catencoder', preprocessors.CatNormalizer(normalizer = OrdinalEncoder().set_output(transform= "pandas"))) 
    #     ])
    
    # prep_pipeline, model = train_pipeline_and_model(X, y)

    # X_transformed = prep_pipeline.fit_transform(X)
    
    # model = LinearRegression().fit(X_transformed, y)

    #Inference
    # y_pred = model.predict(X_transformed)
    # y_pred = pd.DataFrame(y_pred, columns=['y_pred'], index=X.index)

    # y_pred = y_pred.join(fold_training_data[['kfold']])
    # print(y_pred)


    #train model folds here
    # for fold in folds:
        
    #     train_set = X[X['kfold'] != fold]
    #     validation_set = X[X['kfold'] == fold]

        

