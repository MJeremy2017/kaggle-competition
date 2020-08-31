import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

calender = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

sales_train_eval = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv")

event_cnt_1 = len(calender) - sum(calender["event_type_1"].isnull())
event_cnt_2 = len(calender) - sum(calender["event_type_2"].isnull())
print(event_cnt_1, event_cnt_2)

cols = ["d_"+str(i) for i in range(1, 1942)]
# EDA
day2date = calender[['d', 'date']].set_index('d').T.to_dict('list')


def plot_item_sale():
    n = np.random.randint(len(sales_train_eval))
    row = sales_train_eval.iloc[n]
    id = row['id']
    values = row[cols]

    x = [datetime.strptime(day2date.get(i)[0], '%Y-%m-%d') for i in cols]
    plt.figure(figsize=[15, 5])
    plt.plot(x, values)
    plt.title("item {}".format(id))


s = pd.DataFrame(sales_train_eval.iloc[2][cols]).reset_index().rename(columns={'index': 'd', 2: 'count'}).merge(calender, on='d', how='left')
s['count'] = s['count'].astype(int)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
s.groupby('wday').agg({'count': 'mean'}).plot(kind='line', title='sold count/weekday', ax=ax1)
s.groupby('month').agg({'count': 'mean'}).plot(kind='line', title='sold count/month', ax=ax2)
s.groupby('year').agg({'count': 'mean'}).plot(kind='line', title='sold count/year', ax=ax3)