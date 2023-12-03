import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

def grid_search(df_train: pd.DataFrame, df_validation: pd.DataFrame):
    ps = [0, 1, 2, 3, 4, 5]
    qs = [0, 1, 2, 3, 4, 5]
    ls = [0, 1, 2]

    y_train = df.values.reshape(-1)
    y_validation = df_validation.values.reshape(-1)

    pq = zip(ps, qs, ls)

    best_pql = None
    best_aic = 10000000

    for p, q, l in pq:
        model = ARIMA(y_train, order=(p, l, q))
        model = model.fit()
        predict = model.predict(y_validation)

        aic = model.aic
        if aic < best_aic:
            best_aic = aic
            best_pql = (p, q, l)

    return best_pql

df = pd.read_excel("hw2/unemployment.xlsx")
df = df.iloc[:116]

df["Tarih"] = df["Tarih"].apply(lambda x: datetime.strptime(x, "%Y-%m"))

df.set_index("Tarih", inplace=True)
df = df.dropna()

train_len = int(len(df) * 0.7)
len_val = int(len(df) * 0.1)

df_train = df[:train_len]
df_validation = df[train_len:train_len + len_val]
df_test = df[train_len + len_val:]

p, q, l = grid_search(df_train, df_validation)

model = ARIMA(df_test, order=(p, l, q))
model = model.fit()
print(model.summary())
