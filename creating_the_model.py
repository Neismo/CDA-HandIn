import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import warnings
import math as m
from pykliep import DensityRatioEstimator
sns.set_style('white')

# load
data_tr = pd.read_csv(r"data/case1Data.csv")

#data_tr = data_tr.fillna(data_tr.median())

# split
y_tr = data_tr.get('y').to_numpy()
X_tr = data_tr.loc[:].to_numpy()[:, 1:]
C_tr = data_tr.loc[:].to_numpy(dtype=int)[:, 96:]

# clean -> next idea: draw values from normal distribution, with feature's estimated mean and stdev
nan_id_tr = np.array(np.where(np.isnan(X_tr)))
feature_means = np.nanmean(X_tr, axis=0)
feature_stds = np.nanstd(X_tr, axis=0)

#print(feature_means, feature_stds)
for nan_element in nan_id_tr.T:
    row_id, col_id = nan_element
    X_tr[row_id, col_id] = np.random.normal(loc=feature_means[col_id], scale=feature_stds[col_id])



data_tst = pd.read_csv(r"data/case1Data_Xnew.csv")
X_tst = data_tst.loc[:].to_numpy()[:, :]
C_tst = data_tst.loc[:].to_numpy(dtype=int)[:, 95:]

nan_id_tst = np.array(np.where(np.isnan(X_tst)))
for nan_element in nan_id_tst.T:
    row_id, col_id = nan_element
    X_tst[row_id, col_id] = np.random.normal(loc=feature_means[col_id], scale=feature_stds[col_id])

# use elastic net to find best fit
alphas = np.logspace(-1, 4, 200)
l1_ratios = [0.95, 0.99, 1]
K_folds = 5

with warnings.catch_warnings():
    warnings.simplefilter("ignore"),
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=K_folds, n_jobs=-1).fit(X_tr, y_tr)
    )
    model.fit(X_tr, y_tr)
elastic = model.named_steps["elasticnetcv"]

training_fit = model.predict(X_tr)

residuals = y_tr - training_fit

fig, axs = plt.subplots(1, 2)

axs[0].scatter(y_tr, residuals)
axs[1].scatter(training_fit, y_tr)
axs[1].plot(training_fit, training_fit, 'r--', alpha=0.5)
plt.show()

# 1 SE rule
l1_ratio_opt = elastic.l1_ratio_
l1_ratio_opt_id = np.argwhere(np.abs(l1_ratios - l1_ratio_opt) < 10e-5)[0][0]
coefs_opt = elastic.coef_
mses = elastic.mse_path_[l1_ratio_opt_id]
print(f'Non zero coeffients: {len(np.nonzero(coefs_opt)[0])}')

mean_rmse = np.sqrt(np.mean(mses, axis=1))
min_rmse_id = np.argmin(mean_rmse)
min_rmse = mean_rmse[min_rmse_id]
alpha_opt = alphas[min_rmse_id]


SEs = np.std(np.sqrt(mses), axis=1)/np.sqrt(K_folds)

threshold = min_rmse + SEs[min_rmse_id]
one_se_id = np.max(np.argwhere(mean_rmse < threshold))
max_alpha = alphas[one_se_id]

plt.errorbar(alphas, mean_rmse, yerr=SEs, color='navy', ecolor='lightsteelblue', label='Standard Errors')
plt.axvline(max_alpha, color='green', linestyle='-.', alpha=0.4, label='1SE $\\lambda$')
plt.axhline(threshold, color='red', linestyle='-.', alpha=0.4, label='threshold')
plt.axhline(min_rmse, color='blue', linestyle='-.', alpha=0.4, label='Standard Errors')

plt.legend(loc='best')
plt.semilogx()
plt.tight_layout()
plt.grid()

plt.show()
print(f'optimal l1_ratio: {l1_ratio_opt},\n'
      f' optimal lambda: {alpha_opt}, 1SE max lambda: {max_alpha},\n '
      f'min RMSE = {min_rmse}, 1SE RMSE = {mean_rmse[one_se_id]}')

# Get OUT-OF-FOLD predictions — not lasso.predict(X_train)
oof_preds = cross_val_predict(elastic, X_tr, y_tr, cv=5)

# Correct residuals
residuals = y_tr - oof_preds  # ← these are honest
kliep = DensityRatioEstimator(sigmas=[max_alpha])
kliep.fit(X_tr, X_tst)  # keyword arguments are X_train and X_test
w = kliep.predict(X_tr)
w_norm = w / w.sum()
rmse_estimate = np.sqrt(np.sum(w_norm * residuals**2))
print(rmse_estimate)

# 1. Check effective sample size
ess = np.sum(w)**2 / np.sum(w**2)
print(f"ESS: {ess:.1f} out of {len(w)} training points")
# Rule of thumb: ESS < 0.1 * n is a red flag

# 2. Check weight distribution
print(f"Max weight: {w.max():.2f}, Median: {np.median(w):.2f}")
# Max >> median suggests instability

# 3. Compare weighted vs unweighted RMSE from same residuals
rmse_unweighted = np.sqrt(np.mean(residuals**2))
rmse_weighted   = np.sqrt(np.sum(w_norm * residuals**2))
print(f"Ratio: {rmse_unweighted / rmse_weighted:.2f}")
# Ratio >> 1.5 or so warrants investigation