# credit https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50

from datetime import datetime, timedelta
import gc
import numpy as np, pandas as pd
import lightgbm as lgb

CAL_DTYPES = {"event_name_1": "category", "event_name_2": "category", "event_type_1": "category",
              "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
              "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32'}
PRICE_DTYPES = {"store_id": "category", "item_id": "category", "wm_yr_wk": "int16", "sell_price": "float32"}

h = 28
max_lags = 57
tr_last = 1913
fday = datetime(2016, 4, 25)

is_train = True
nrows = None
first_day = 1200

prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv", dtype=PRICE_DTYPES)
for col, col_dtype in PRICE_DTYPES.items():
    if col_dtype == "category":
        prices[col] = prices[col].cat.codes.astype("int16")
        prices[col] -= prices[col].min()
prices.head()

cal = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv", dtype=CAL_DTYPES)
cal["date"] = pd.to_datetime(cal["date"])
for col, col_dtype in CAL_DTYPES.items():
    if col_dtype == "category":
        cal[col] = cal[col].cat.codes.astype("int16")
        cal[col] -= cal[col].min()
cal.head()

# take only past 2 year data
start_day = max(1 if is_train else tr_last - max_lags, first_day)
print("start_day", start_day)
print("last day", tr_last)

numcols = [f"d_{day}" for day in range(start_day, tr_last + 1)]
catcols = ['id', 'item_id', 'dept_id', 'store_id', 'cat_id', 'state_id']
dtype = {numcol: "float32" for numcol in numcols}
dtype.update({col: "category" for col in catcols if col != "id"})
dt = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv",
                 nrows=nrows, usecols=catcols + numcols, dtype=dtype)

for col in catcols:
    if col != "id":
        dt[col] = dt[col].cat.codes.astype("int16")
        dt[col] -= dt[col].min()
dt.head()

if not is_train:
    for day in range(tr_last + 1, tr_last + 28 + 1):
        dt[f"d_{day}"] = np.nan

# unpivot dt
dt = pd.melt(dt,
             id_vars=catcols,
             value_vars=[col for col in dt.columns if col.startswith("d_")],
             var_name="d",
             value_name="sales")
# get each item at a specific day with features + sales
dt = dt.merge(cal, on="d", copy=False)
dt = dt.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], copy=False)
dt.head()

lags = [7, 28]
lag_cols = [f"lag_{lag}" for lag in lags]
print(lag_cols)
# sales last 7 and 28 days ago
for lag, lag_col in zip(lags, lag_cols):
    dt[lag_col] = dt[["id", "sales"]].groupby("id")["sales"].shift(lag)
dt.head()

wins = [7, 28]
for win in wins:
    for lag, lag_col in zip(lags, lag_cols):
        dt[f"rmean_{lag}_{win}"] = dt[["id", lag_col]].groupby("id")[lag_col].transform(lambda x: x.rolling(win).mean())
dt.head()

# adding date feature
date_features = {
    "wday": "weekday",
    "week": "weekofyear",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
    "mday": "day"
}

for date_feat_name, date_feat_func in date_features.items():
    if date_feat_name in dt.columns:
        dt[date_feat_name] = dt[date_feat_name].astype("int16")
    else:
        # The getattr() method returns the value of the named attribute of an object
        dt[date_feat_name] = getattr(dt["date"].dt, date_feat_func).astype("int16")
