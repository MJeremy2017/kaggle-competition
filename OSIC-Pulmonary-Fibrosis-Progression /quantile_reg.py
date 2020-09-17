import numpy as np
import random
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
from timeit import timeit
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold

# color
from colorama import Fore, Back, Style

import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


train_df = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
train_df.sample(5)