import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.metrics import f1_score


def get_number_in_str(s: str):
    return [int(_) for _ in s.split(" ") if _.isdigit()][0]


def convert_to_timestamp(date_string):
    date_object = datetime.strptime(date_string, "%b-%Y")

    timestamp = int(date_object.timestamp())

    return timestamp

def apply_5_fold(df: pd.DataFrame, l: float, penalty: str):
    ls = [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]

    fold_length = len(df) // 5

    scores = {}

    for i in range(5):
        df_test = df.loc[i * fold_length: (i + 1) * fold_length]
        df_train = pd.concat([df.loc[:i * fold_length], df.loc[(i + 1) * fold_length:]])

        X_test = df_test[df.columns[:-1]].values
        y_test = df_test["home_ownership"].values

        X_train = df_train[df.columns[:-1]].values
        y_train = df_train["home_ownership"].values

        reg = LogisticRegression(penalty=penalty, max_iter=100, solver="saga").fit(X_train, y_train)
        test_predict = reg.predict(X_test)

        scores[i] = f1_score(y_test, test_predict, average='weighted')

    return np.mean(list(scores.values()))

df = pd.read_csv("hw2/short_lending_data.csv")

df_x = df[df.columns[df.columns != "home_ownership"]]
df_y = df["home_ownership"]

df_x = df_x.select_dtypes(include='number')
df = pd.concat([df_x, df_y], axis=1)

df = df.dropna(axis=0)

for l in [0.001, 0.01, 0.1, 0.25, 0.5, 1.0]:
    rmse = apply_5_fold(df, l , "l1")
    print("L1", l, rmse)

    rmse = apply_5_fold(df, l, "l2")
    print("L2", l, rmse)

    print()