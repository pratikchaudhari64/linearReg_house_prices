import pandas as pd
from train import train_pipeline_and_model

def predict(X_train, y_train, X_pred):

    prep_pipeline, model = train_pipeline_and_model(X_train, y_train)

    X_pred_transformed = prep_pipeline.fit_transform(X_pred)

    y_pred = model.predict(X_pred_transformed)

    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])

    return y_pred


if __name__ == "__main__":

    fold_training_data = pd.read_csv("./inputs/train_folds.csv")
    fold_training_data.set_index('Id', inplace=True)
    testset = pd.read_csv("./inputs/test.csv")
    testset.set_index('Id', inplace=True)

    y_train = fold_training_data[['SalePrice']]
    X_train = fold_training_data.drop(['SalePrice', 'kfold'], axis = 1)

    y_pred = predict(X_train, y_train, X_pred=testset)

    print(y_pred)