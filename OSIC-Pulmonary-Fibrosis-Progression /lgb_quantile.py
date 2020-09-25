import numpy as np
import pandas as pd
import pydicom
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

ROOT = "../input/osic-pulmonary-fibrosis-progression"
tr = pd.read_csv(f"{ROOT}/train.csv")
tr.drop_duplicates(keep=False, inplace=True, subset=['Patient', 'Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")

print("add infos")
sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub = sub[['Patient', 'Weeks', 'Confidence', 'Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")

tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])

data['min_week'] = data['Weeks']
data.loc[data.WHERE == 'test', 'min_week'] = np.nan
data['min_week'] = data.groupby('Patient')['min_week'].transform('min')

base = data.loc[data.Weeks == data.min_week]
base = base[['Patient', 'FVC']].copy()
base.columns = ['Patient', 'min_FVC']
base['nb'] = 1
base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')
base = base[base.nb == 1]
base.drop('nb', axis=1, inplace=True)

data = data.merge(base, on='Patient', how='left')
data['base_week'] = data['Weeks'] - data['min_week']
del base

for col in ['Sex', 'SmokingStatus']:
    data[col] = data[col].astype('category').cat.codes

feature_list = ['Age', 'Sex', 'SmokingStatus', 'Percent', 'base_week', 'min_FVC']
cat_feat = ['Sex', 'SmokingStatus']

tr = data.loc[data.WHERE == 'train']
chunk = data.loc[data.WHERE == 'val']
sub = data.loc[data.WHERE == 'test']
del data

lgb_params = {
    'n_jobs': 1,
    'max_depth': 4,
    'min_data_in_leaf': 16,
    'subsample': 0.9,
    'n_estimators': 500,
    'learning_rate': 0.02,
    'colsample_bytree': 0.9,
    'boosting_type': 'gbdt',
    'metric': ['quantile', 'rmse']
}

y = tr['FVC']  # .values
z = tr[feature_list]  # .values
ze = sub[feature_list]  # .values

NFOLD = 5
kf = KFold(n_splits=NFOLD)

pred = np.zeros((z.shape[0], 3))
pe = np.zeros((ze.shape[0], 3))

quantiles = [0.2, 0.5, 0.8]
cnt = 0
for tr_idx, val_idx in kf.split(z):
    cnt += 1
    for i in range(len(quantiles)):
        q = quantiles[i]
        print(f"FOLD {cnt}, quantile {q}")
        lgb = LGBMRegressor(objective='quantile', alpha=q, **lgb_params)
        lgb.fit(X=z.loc[tr_idx], y=y[tr_idx], eval_set=[[z.loc[val_idx], y[val_idx]]],
                categorical_feature=cat_feat, early_stopping_rounds=10, verbose=0)

        pred[val_idx, i] = lgb.predict(z.loc[val_idx])
        pe[:, i] += lgb.predict(ze) / NFOLD

err = mean_absolute_error(y, pred[:, 1])
unc = np.mean(pred[:, 2] - pred[:, 0])
print(err, unc)


def get_submission(sub, pe):
    sub['FVC1'] = pe[:, 1] * 0.996
    sub['Confidence1'] = pe[:, 2] - pe[:, 0]

    subm = sub[['Patient_Week', 'FVC', 'Confidence', 'FVC1', 'Confidence1']].copy()
    subm.loc[~subm.FVC1.isnull(), 'FVC'] = subm.loc[~subm.FVC1.isnull(), 'FVC1']
    subm.loc[~subm.FVC1.isnull(), 'Confidence'] = subm.loc[~subm.FVC1.isnull(), 'Confidence1']

    print("fill in prediction that already exists")
    otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
    for i in range(len(otest)):
        subm.loc[subm['Patient_Week'] == otest.Patient[i] + '_' + str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]
        subm.loc[subm['Patient_Week'] == otest.Patient[i] + '_' + str(otest.Weeks[i]), 'Confidence'] = 0.1
    subm[["Patient_Week", "FVC", "Confidence"]].to_csv("submission.csv", index=False)
    print("sub file saved")


get_submission(sub, pe)
