import numpy as np
import pandas as pd
import os
import pickle
import sys
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression

# [TradeMaster] dataset is argv[1] (was hard-coded ETHUSDT); run.sh passes $DATASET.
# Bar-count constants (upstream: 1-minute bars) can be rescaled for coarser bars through the environment:
#   MACRO_CHUNK_SIZE (4320 = 3 days of minutes): market-type chunks for the sub-agents
#   MACRO_CONTEXT_WINDOW (360): rolling slope_<w> / vol_<w> context features of the hyper-agent
DATA = os.path.join('./data', sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else 'ETHUSDT')
CHUNK_SIZE = int(os.environ.get("MACRO_CHUNK_SIZE", 4320))
CONTEXT_WINDOW = int(os.environ.get("MACRO_CONTEXT_WINDOW", 360))

def smooth_data(data):
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    return filtfilt(b, a, data)

def get_slope(smoothed_data):
    X = np.arange(len(smoothed_data)).reshape(-1, 1)
    model = LinearRegression().fit(X, smoothed_data)
    return model.coef_[0]

def get_slope_window(window):
    N, Wn = 1, 0.05
    b, a = butter(N, Wn, btype='low')
    y = filtfilt(b, a, window.values)
    X = np.arange(len(y)).reshape(-1, 1)   
    model = LinearRegression().fit(X, y)
    return model.coef_[0]

def chunk(df_train, df_val, df_test):
    chunk_size = CHUNK_SIZE  # [TradeMaster] was 4320
    for i in range(int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_train[start:end].reset_index(drop=True)
        df_chunk.to_feather(os.path.join(DATA, 'train', 'df_{}.feather'.format(i)))

    for i in range(int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_val[start:end].reset_index(drop=True)
        df_chunk.to_feather(os.path.join(DATA, 'val', 'df_{}.feather'.format(i)))

    for i in range(int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        df_chunk = df_test[start:end].reset_index(drop=True)
        df_chunk.to_feather(os.path.join(DATA, 'test', 'df_{}.feather'.format(i)))

def label_slope(df_train, df_val, df_test):
    chunk_size = CHUNK_SIZE  # [TradeMaster] was 4320
    slopes_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_train.append(slope)

    slopes_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_val.append(slope)

    slopes_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test['close'][start:end].values
        smoothed_chunk = smooth_data(chunk)
        slope = get_slope(smoothed_chunk)
        slopes_test.append(slope)

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    slope_labels_train, bins = pd.qcut(slopes_train, q=quantiles, retbins=True, labels=False)

    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(slope_labels_train):
        train_indices[label].append(index)
    with open(os.path.join(DATA, 'train', 'slope_labels.pkl'), 'wb') as file:
        pickle.dump(train_indices, file)

    # [TradeMaster] open-ended outer bins (were -100/100: a pair with larger slopes got NaN labels and crashed)
    bins[0] = -np.inf
    bins[-1] = np.inf
    slope_labels_val = pd.cut(slopes_val, bins=bins, labels=False, include_lowest=True)
    slope_labels_val = [1 if element == 0 else element for element in slope_labels_val]
    slope_labels_val = [3 if element == 4 else element for element in slope_labels_val]
    slope_labels_test = pd.cut(slopes_test, bins=bins, labels=False, include_lowest=True)
    slope_labels_test = [1 if element == 0 else element for element in slope_labels_test]
    slope_labels_test = [3 if element == 4 else element for element in slope_labels_test]

    for index, label in enumerate(slope_labels_val):
        val_indices[label].append(index)
    with open(os.path.join(DATA, 'val', 'slope_labels.pkl'), 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(slope_labels_test):
        test_indices[label].append(index)
    with open(os.path.join(DATA, 'test', 'slope_labels.pkl'), 'wb') as file:
        pickle.dump(test_indices, file)

def label_volatility(df_train, df_val, df_test):
    chunk_size = CHUNK_SIZE  # [TradeMaster] was 4320
    volatilities_train = []
    for i in range(0, int(len(df_train) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_train[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_train.append(volatility)

    volatilities_val = []
    for i in range(0, int(len(df_val) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_val[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_val.append(volatility)
    
    volatilities_test = []
    for i in range(0, int(len(df_test) / chunk_size)):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = df_test[start:end]
        chunk['return'] = chunk['close'].pct_change().fillna(0)
        volatility = chunk['return'].std()
        volatilities_test.append(volatility)

    quantiles = [0, 0.05, 0.35, 0.65, 0.95, 1]
    vol_labels_train, bins = pd.qcut(volatilities_train, q=quantiles, retbins=True, labels=False)

    train_indices = [[] for _ in range(5)]
    val_indices = [[] for _ in range(5)]
    test_indices = [[] for _ in range(5)]
    for index, label in enumerate(vol_labels_train):
        train_indices[label].append(index)
    with open(os.path.join(DATA, 'train', 'vol_labels.pkl'), 'wb') as file:
        pickle.dump(train_indices, file)

    # [TradeMaster] open-ended outer bins (were 0/1)
    bins[0] = -np.inf
    bins[-1] = np.inf
    vol_labels_val = pd.cut(volatilities_val, bins=bins, labels=False, include_lowest=True)
    vol_labels_val = [1 if element == 0 else element for element in vol_labels_val]
    vol_labels_val = [3 if element == 4 else element for element in vol_labels_val]
    vol_labels_test = pd.cut(volatilities_test, bins=bins, labels=False, include_lowest=True)
    vol_labels_test = [1 if element == 0 else element for element in vol_labels_test]
    vol_labels_test = [3 if element == 4 else element for element in vol_labels_test]

    for index, label in enumerate(vol_labels_val):
        val_indices[label].append(index)
    with open(os.path.join(DATA, 'val', 'vol_labels.pkl'), 'wb') as file:
        pickle.dump(val_indices, file)
    for index, label in enumerate(vol_labels_test):
        test_indices[label].append(index)
    with open(os.path.join(DATA, 'test', 'vol_labels.pkl'), 'wb') as file:
        pickle.dump(test_indices, file)

def slope_weights(window_size):
    # [TradeMaster] filtfilt (odd padding) and the least-squares slope are both linear in the window,
    # so get_slope_window(window) == window @ w with w[i] = get_slope_window(e_i). Computed once per size.
    return np.array([get_slope_window(pd.Series(e)) for e in np.eye(window_size)])


def rolling_slope(close, window_size):
    # [TradeMaster] replaces close.rolling(window_size).apply(get_slope_window): one butter/filtfilt/
    # LinearRegression fit per row -> one dot product per row. Equal to ~1e-9 relative (float rounding).
    close = np.asarray(close, dtype=np.float64)
    out = np.full(len(close), np.nan)
    if len(close) >= window_size:
        windows = np.lib.stride_tricks.sliding_window_view(close, window_size)
        out[window_size - 1:] = windows @ slope_weights(window_size)
    return out


def label_whole(df):
    window_size_list = [CONTEXT_WINDOW]  # [TradeMaster] was [360]
    for i in range(len(window_size_list)):
        window_size = window_size_list[i]
        df['slope_{}'.format(window_size)] = rolling_slope(df['close'], window_size)  # [TradeMaster] was rolling().apply
        df['return'] = df['close'].pct_change().fillna(0)
        df['vol_{}'.format(window_size)] = df['return'].rolling(window=window_size).std()
    return df

if __name__ == "__main__":
    df_train = pd.read_feather(os.path.join(DATA, 'df_train.feather'))
    df_val = pd.read_feather(os.path.join(DATA, 'df_val.feather'))
    df_test = pd.read_feather(os.path.join(DATA, 'df_test.feather'))

    os.makedirs(os.path.join(DATA, 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'val'), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'test'), exist_ok=True)
    os.makedirs(os.path.join(DATA, 'whole'), exist_ok=True)

    chunk(df_train, df_val, df_test)
    label_slope(df_train, df_val, df_test)
    label_volatility(df_train, df_val, df_test)

    df_train = label_whole(df_train).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_val = label_whole(df_val).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)
    df_test = label_whole(df_test).dropna().reset_index(drop=True).iloc[1:].reset_index(drop=True)

    df_train.to_feather(os.path.join(DATA, 'whole', 'train.feather'))
    df_val.to_feather(os.path.join(DATA, 'whole', 'val.feather'))
    df_test.to_feather(os.path.join(DATA, 'whole', 'test.feather'))


    