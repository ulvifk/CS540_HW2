import datetime
import pickle

import gurobipy as gp
import numpy as np
import pandas as pd
import yfinance as yf
from gurobipy import GRB
from matplotlib import pyplot as plt

constituent_tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]["Symbol"].unique()


def fetch_data():
    closing_price_mapping_train = {}
    closing_price_mapping_test = {}

    start_epoch = datetime.datetime(2015, 1, 1).timestamp()
    for ticker_name in constituent_tickers:
        ticker = yf.Ticker(ticker_name)

        try:
            first_trade_date = ticker.history_metadata["firstTradeDate"]
            if first_trade_date > start_epoch:
                continue
        except:
            continue

        closing_price_mapping_train[ticker_name] = yf.download(ticker_name, start="2015-01-01", end="2015-12-31")[
            "Close"]
        closing_price_mapping_test[ticker_name] = yf.download(ticker_name, start="2016-01-01", end="2016-12-31")[
            "Close"]

    with open("closing_price_mapping_train.pickle", "wb") as f:
        pickle.dump(closing_price_mapping_train, f)

    with open("closing_price_mapping_test.pickle", "wb") as f:
        pickle.dump(closing_price_mapping_test, f)


def fetch_sp500_data():
    test_sp500 = yf.download("^GSPC", start="2016-01-01", end="2016-12-31")[
        "Close"]
    test_df: pd.DataFrame
    test_df = pd.DataFrame(test_sp500)
    test_df["returns"] = test_df["Close"].pct_change()
    test_df = test_df.dropna()
    test_df["cumulative_returns"] = (test_df["returns"] + 1).cumprod()

    test_df.to_pickle("sp500_test.pickle")


def create_model(means: np.array, covariance_matrix: np.ndarray, alpha: float) -> tuple[gp.Model, gp.Var]:
    m = gp.Model("MVP")

    w = m.addMVar(shape=(len(means),), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w")

    obj = w.T @ means - alpha * w.T @ covariance_matrix @ w
    m.setObjective(obj, GRB.MAXIMIZE)

    budget_constraint = gp.quicksum(w)
    m.addConstr(budget_constraint == 1, name="budget")

    return m, w

def make_psd(matrix: np.ndarray) -> np.ndarray:
    if np.all(np.linalg.eigvals(matrix) > 0):
        return matrix
    else:
        return matrix + 1e-6 * np.identity(matrix.shape[0])

def train_model(df_train: pd.DataFrame, alpha: float) -> np.array:
    X_train = df_train.values

    means = np.mean(X_train, axis=0)
    covariance_matrix = np.cov(X_train.T)
    covariance_matrix = make_psd(covariance_matrix)

    m, w = create_model(means, covariance_matrix, alpha)
    m.optimize()
    w = w.x
    m.dispose()

    return w

def simulation(df_train: pd.DataFrame, df_test: pd.DataFrame, alpha: float):
    df_returns = pd.DataFrame(columns= ["return"])

    current_w = train_model(df_train, alpha)

    df_test = df_test.pct_change().dropna()

    for index, row in df_test.iterrows():
        if index.day_of_week == 0:
            current_w = train_model(df_train.loc[:index], alpha)

        current_return = current_w @ row.values
        df_returns.loc[index] = current_return

        df_train.loc[index] = row
        df_test = df_test.drop(index)

    df_returns["cumulative_return"] = (df_returns["return"] + 1).cumprod()

    return df_returns

def plot_return_graph(df_returns: dict[int, pd.DataFrame], df_sp500: pd.DataFrame):
    fig, ax = plt.subplots()
    fig: plt.Figure
    ax: plt.Axes

    for alpha, df_returns in df_returns.items():
        ax.plot(df_returns["cumulative_return"], label=f"Alpha {alpha}", linewidth=2)

    ax.plot(df_sp500["cumulative_returns"], label="S&P 500", linewidth=2)

    # Set the title and labels
    ax.set_title('Cumulative Return Over Time', fontsize=16)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)

    # Customize the grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add a legend
    ax.legend(loc='upper left')

    # Customize the tick labels and style
    ax.tick_params(axis='both', which='both', labelsize=12)
    plt.xticks(rotation=45)

    fig.show()

def calculate_IRs(df_returns: pd.DataFrame, df_sp500: pd.DataFrame):
    excess_returns = df_returns["portfolio_return"] - df_sp500["returns"]
    tracking_error = np.std(excess_returns, ddof=1)
    return np.mean(excess_returns) / tracking_error

#fetch_sp500_data()
#fetch_data()

df_sp500 = pd.read_pickle("sp500_test.pickle")

with open("closing_price_mapping_train.pickle", "rb") as f:
    closing_price_mapping_train = pickle.load(f)

with open("closing_price_mapping_test.pickle", "rb") as f:
    closing_price_mapping_test = pickle.load(f)

df_train = pd.DataFrame(closing_price_mapping_train)
X_train = df_train.values

df_test = pd.DataFrame(closing_price_mapping_test)

returns = {}

for alpha in [0.1, 1, 5, 10]:
    df_returns = simulation(df_train, df_test, alpha)
    returns[alpha] = df_returns

plot_return_graph(returns, df_sp500)

for alpha, _returns in returns.items():
    IR = calculate_IRs(_returns, df_sp500)
    print(f"IR for alpha {alpha}: {IR}")