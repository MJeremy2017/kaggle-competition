# *Numerical Columns: * Depending on your environment, pandas automatically creates int32, int64, float32 or float64
# columns for numeric ones. If you know the min or max value of a column, you can use a subtype which is less memory
# consuming. You can also use an unsigned subtype if there is no negative value. Here are the different subtypes you
# can use: int8 / uint8 : consumes 1 byte of memory, range between -128/127 or 0/255 bool : consumes 1 byte,
# true or false float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
# float32 / int32 / uint32 : consumes 4 bytes of memory, range between -2147483648 and 2147483647 float64 / int64 /
# uint64: consumes 8 bytes of memory If one of your column has values between 1 and 10 for example, you will reduce
# the size of that column from 8 bytes per row to 1 byte, which is more than 85% memory saving on that column!

# *Categorical Columns: * Pandas stores categorical columns as objects. One of the reason this storage is not optimal
# is that it creates a list of pointers to the memory address of each value of your column. For columns with low
# cardinality (the amount of unique values is lower than 50% of the count of these values), this can be optimized by
# forcing pandas to use a virtual mapping table where all unique values are mapped via an integer instead of a
# pointer. This is done using the category datatype.

# https://www.kaggle.com/anshuls235/time-series-forecasting-eda-fe-modelling
import os
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings

warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor
import joblib

sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales.name = 'sales'
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
calendar.name = 'calendar'
prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
prices.name = 'prices'

# Add zero sales for the remaining days 1942-1969
for d in range(1942, 1970):
    col = 'd_' + str(d)
    sales[col] = 0
    sales[col] = sales[col].astype(np.int16)
sales.head()

sales_bd = np.round(sales.memory_usage().sum() / (1024 * 1024), 1)
calendar_bd = np.round(calendar.memory_usage().sum() / (1024 * 1024), 1)
prices_bd = np.round(prices.memory_usage().sum() / (1024 * 1024), 1)
print(sales_bd)


# Downcast in order to save memory
def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, t in enumerate(types):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df


sales = downcast(sales)
prices = downcast(prices)
calendar = downcast(calendar)

sales_ad = np.round(sales.memory_usage().sum() / (1024 * 1024), 1)
calendar_ad = np.round(calendar.memory_usage().sum() / (1024 * 1024), 1)
prices_ad = np.round(prices.memory_usage().sum() / (1024 * 1024), 1)
print(sales_ad)

dic = {'DataFrame': ['sales', 'calendar', 'prices'],
       'Before downcasting': [sales_bd, calendar_bd, prices_bd],
       'After downcasting': [sales_ad, calendar_ad, prices_ad]}

memory = pd.DataFrame(dic)
memory = pd.melt(memory, id_vars='DataFrame', var_name='Status', value_name='Memory (MB)')
memory.sort_values('Memory (MB)', inplace=True)
fig = px.bar(memory, x='DataFrame', y='Memory (MB)', color='Status', barmode='group', text='Memory (MB)')
fig.update_traces(texttemplate='%{text} MB', textposition='outside')
fig.update_layout(template='seaborn', title='Effect of Downcasting')
fig.show()

# melting data
df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d',
             value_name='sold').dropna()
df = pd.merge(df, calendar, on='d', how='left')
df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# tree map
group = sales.groupby(['state_id', 'store_id', 'cat_id', 'dept_id'], as_index=False)['item_id'].count().dropna()
group['USA'] = 'United States of America'
group.rename(columns={'state_id': 'State', 'store_id': 'Store', 'cat_id': 'Category', 'dept_id': 'Department',
                      'item_id': 'Count'}, inplace=True)
fig = px.treemap(group, path=['USA', 'State', 'Store', 'Category', 'Department'], values='Count',
                 color='Count',
                 color_continuous_scale=px.colors.sequential.Sunset,
                 title='Walmart: Distribution of items')
fig.update_layout(template='seaborn')
fig.show()
