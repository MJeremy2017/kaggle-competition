import numpy as np
import pandas as pd

calender = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

sales_train_eval = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv")

event_cnt_1 = len(calender) - sum(calender["event_type_1"].isnull())
event_cnt_2 = len(calender) - sum(calender["event_type_2"].isnull())
print(event_cnt_1, event_cnt_2)