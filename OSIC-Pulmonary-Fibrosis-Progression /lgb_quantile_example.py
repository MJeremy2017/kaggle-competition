# http://ethen8181.github.io/machine-learning/ab_tests/quantile_regression/quantile_regression.html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


def compute_quantile_loss(y_true, y_pred, quantile):
    """

    Parameters
    ----------
    y_true : 1d ndarray
        Target value.

    y_pred : 1d ndarray
        Predicted value.

    quantile : float, 0. ~ 1.
        Quantile to be evaluated, e.g., 0.5 for median.
    """
    residual = y_true - y_pred
    return np.maximum(quantile * residual, (quantile - 1) * residual)


n_samples = 1000
y_true = np.zeros(n_samples)
y_pred = np.linspace(-1, 1, n_samples)

quantiles = [0.1, 0.5, 0.9]
quantile_losses = [
    compute_quantile_loss(y_true, y_pred, quantile) for quantile in quantiles
]

plt.rcParams['figure.figsize'] = 12, 8
plt.rcParams['font.size'] = 12

for quantile_loss in quantile_losses:
    plt.plot(y_pred, quantile_loss)

plt.legend([str(int(q * 100)) + 'th percentile' for q in quantiles])
plt.xlabel('Error')
plt.ylabel('Quantile loss')
plt.title('Quantile loss by error and quantile', loc='left')
plt.show()


def ground_truth(x):
    """Ground truth -- function to approximate"""
    return x * np.sin(x) + np.sin(2 * x)


def gen_data(low, high, n_samples, scale=4, test_size=0.2, random_state=3):
    """generate training and testing data from the ground truth function"""
    np.random.seed(15)
    X = np.random.uniform(low, high, size=n_samples)

    # generate the response from the ground truth function and add
    # some random noise to it, scale controls the variance of the noise.
    y = ground_truth(X) + np.random.normal(scale=scale, size=n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def plot_data(x_plot, X_train, X_test, y_train, y_test, low, high):
    """plot training and testing data"""
    s = 15
    plt.plot(x_plot, ground_truth(x_plot), alpha=0.5, label='ground truth')
    plt.scatter(X_train, y_train, s=s, alpha=0.2)
    plt.scatter(X_test, y_test, s=s, alpha=0.2, color='red')
    plt.xlim((low, high))
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend(loc='upper left')
    plt.show()


low = 0
high = 20
x_plot = np.linspace(low, high, 500)

n_samples = 2000
X_train, X_test, y_train, y_test = gen_data(low=low, high=high, n_samples=n_samples)
plot_data(x_plot, X_train, X_test, y_train, y_test, low, high)

# feel free to do some hyperparamter tuning,
# these parameters were picked randomly
lgb_params = {
    'n_jobs': 1,
    'max_depth': 4,
    'min_data_in_leaf': 10,
    'subsample': 0.9,
    'n_estimators': 80,
    'learning_rate': 0.1,
    'colsample_bytree': 0.9,
    'boosting_type': 'gbdt'
}
lgb_l2 = LGBMRegressor(objective='regression', **lgb_params)
lgb_l2.fit(X_train[:, np.newaxis], y_train)

plt.plot(x_plot, lgb_l2.predict(x_plot[:, np.newaxis]),
         label='LGB L2 Loss', alpha=0.9, linewidth=2)

plot_data(x_plot, X_train, X_test, y_train, y_test, low, high)

quantile_alphas = [0.1, 0.5, 0.9]

lgb_quantile_alphas = {}
for quantile_alpha in quantile_alphas:
    # to train a quantile regression, we change the objective parameter and
    # specify the quantile value we're interested in
    lgb = LGBMRegressor(objective='quantile', alpha=quantile_alpha, **lgb_params)
    lgb.fit(X_train[:, np.newaxis], y_train)
    lgb_quantile_alphas[quantile_alpha] = lgb

for quantile_alpha, lgb in lgb_quantile_alphas.items():
    plt.plot(x_plot, lgb.predict(x_plot[:, np.newaxis]),
             label='LGB quantile alpha: {}'.format(quantile_alpha),
             alpha=0.9, linewidth=2)

plot_data(x_plot, X_train, X_test, y_train, y_test, low, high)


def create_loss_comparision():
    model_name = []
    columns = []
    results = []

    y_true = ground_truth(x_plot[:, np.newaxis])
    for quantile_alpha, lgb in lgb_quantile_alphas.items():
        quantile_str = str(int(quantile_alpha * 100))
        columns.append('quantile_' + quantile_str)
        model_name.append('lgb_' + quantile_str)

        y_pred = lgb.predict(x_plot[:, np.newaxis])
        result = [
            compute_quantile_loss(y_true, y_pred, quantile).mean()
            for quantile in quantile_alphas
        ]
        results.append(result)

    df_results = pd.DataFrame(results, columns=columns)
    df_results['model'] = model_name
    return df_results


df_results = create_loss_comparision()
df_results


def score_barplot(df_results, model_col='model', figsize=(8, 6)):
    metrics_cols = ['quantile_10', 'quantile_50', 'quantile_90']
    colors = ['lightskyblue', 'lightcoral', 'gold']

    width = 0.3
    ind = np.arange(len(metrics_cols))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    n_rows = df_results.shape[0]
    for i in range(n_rows):
        x = ind + (i * width)
        height = df_results.loc[i, metrics_cols]
        label = df_results.loc[i, model_col]
        ax.bar(x, height, width, label=label, color=colors[i])

    half_width = (n_rows // 2) * width
    ax.set_xticks(ind + half_width)
    ax.set_xticklabels(metrics_cols)
    ax.set_ylabel('quantile loss')
    ax.set_xlabel('quantile')
    ax.legend()
    plt.tight_layout()
    plt.show()


score_barplot(df_results)
