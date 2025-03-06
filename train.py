# get training folds csv.
# describe prep and model pipeline
# write cross validation to pick one fold as validation set and rest as training set
# for each iteration of cross validation, apply pipeline (prep steps + model) on training set and calculate error metrics on validation set

import pandas as pd
import preprocessors

if __name__ == "__main__":

    fold_training_data = pd.read_csv("./inputs/train_folds.csv")
    
    folds = fold_training_data['kfold'].unique()

    for fold in folds:
        
        train_set = fold_training_data[fold_training_data['kfold'] != fold]
        validation_set = fold_training_data[fold_training_data['kfold'] == fold]

        # pipeline of prep steps; and estimator. Need these steps to be set-up in different place. Want the train.py to only pick up whatever steps needed and train the model and return eval metrics
