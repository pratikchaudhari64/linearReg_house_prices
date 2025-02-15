import pandas as pd
from sklearn import model_selection
import os

target = 'SalePrice'


if __name__ == "__main__":

    df = pd.read_csv("./inputs/train.csv")
    
    df["kfold"] = -1

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target].values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv("./inputs/train_folds.csv", index=False)