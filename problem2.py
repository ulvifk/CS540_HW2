from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
import sklearn.metrics as metrics


def apply_5_fold(df: pd.DataFrame, reg_func, l: float):
    ls = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]

    fold_length = len(df) // 5

    rmses = {}

    for i in range(5):
        df_test = df.loc[i * fold_length: (i + 1) * fold_length]
        df_train = pd.concat([df.loc[:i * fold_length], df.loc[(i + 1) * fold_length:]])

        X_test = df_test[df.columns[:-1]].values
        y_test = df_test["SalePrice"].values

        X_train = df_train[df.columns[:-1]].values
        y_train = df_train["SalePrice"].values

        reg = reg_func(alpha=l)
        reg = reg.fit(X_train, y_train)

        test_predict = reg.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, test_predict))
        rmses[i] = rmse

    return np.mean(list(rmses.values()))

df = pd.read_csv("hw2/kaggle_house.csv")
df = df.select_dtypes(include='number')
df = df.dropna(axis=0)

df_train = df[:len(df) // 2]
df_test = df[len(df) // 2:]

X_train = df_train[df.columns[:-1]].values
y_train = df_train["SalePrice"].values

X_test = df_test[df.columns[:-1]].values
y_test = df_test["SalePrice"].values

for l in [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]:
    rmse = apply_5_fold(df, Lasso, l)
    print("Lasso", l, rmse)

    rmse = apply_5_fold(df, Ridge, l)
    print("Ridge", l, rmse)

    print()



