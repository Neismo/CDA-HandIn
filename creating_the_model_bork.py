import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
import seaborn as sns
import warnings

from pykliep import DensityRatioEstimator

sns.set_style('white')

_SEED = 1983202
np.random.seed(_SEED)

# load
data_tr = pd.read_csv(r"data/case1Data.csv")

#data_tr = data_tr.fillna(data_tr.median())

# split
y_tr = data_tr.get('y').to_numpy()
X_tr_raw = data_tr.loc[:].to_numpy()[:, 1:]

# The final five features are categorical and are handled separately.
n_categorical = 5
X_tr_cont = X_tr_raw[:, :-n_categorical].astype(float)
X_tr_cat = X_tr_raw[:, -n_categorical:]

# clean -> next idea: draw values from normal distribution, with feature's estimated mean and stdev
nan_id_tr = np.array(np.where(np.isnan(X_tr_cont)))
feature_means_tr = np.nanmean(X_tr_cont, axis=0)
feature_stds_tr = np.nanstd(X_tr_cont, axis=0)

#print(feature_means, feature_stds)
for nan_element in nan_id_tr.T:
    row_id, col_id = nan_element
    X_tr_cont[row_id, col_id] = np.random.normal(loc=feature_means_tr[col_id], scale=feature_stds_tr[col_id])

data_tst = pd.read_csv(r"data/case1Data_Xnew.csv")
X_tst_raw = data_tst.loc[:].to_numpy() # does not have true y values.
X_tst_cont = X_tst_raw[:, :-n_categorical].astype(float)
X_tst_cat = X_tst_raw[:, -n_categorical:]

nan_id_tst = np.array(np.where(np.isnan(X_tst_cont)))
feature_means_tst = np.nanmean(X_tst_cont, axis=0)
feature_stds_tst = np.nanstd(X_tst_cont, axis=0)

for nan_element in nan_id_tst.T:
    row_id, col_id = nan_element
    X_tst_cont[row_id, col_id] = np.random.normal(loc=feature_means_tst[col_id], scale=feature_stds_tst[col_id])

def one_hot_encode_categorical(train_cat: np.ndarray, test_cat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_blocks = []
    test_blocks = []

    for col_idx in range(train_cat.shape[1]):
        prefix = f"cat{col_idx}"

        train_series = pd.Series(train_cat[:, col_idx], dtype="float")
        train_dummies = pd.get_dummies(train_series, prefix=prefix, dummy_na=False)

        test_series = pd.Series(test_cat[:, col_idx], dtype="float")
        test_dummies = pd.get_dummies(test_series, prefix=prefix, dummy_na=False)
        test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

        train_blocks.append(train_dummies.to_numpy(dtype=float))
        test_blocks.append(test_dummies.to_numpy(dtype=float))

    return np.hstack(train_blocks), np.hstack(test_blocks)


X_tr_cat_ohe, X_tst_cat_ohe = one_hot_encode_categorical(X_tr_cat, X_tst_cat)
X_tr = np.hstack([X_tr_cont, X_tr_cat_ohe])
X_tst = np.hstack([X_tst_cont, X_tst_cat_ohe])

feature_correlation = np.corrcoef(X_tr, rowvar=False)
highest_correlation = np.max(feature_correlation-np.eye(len(feature_correlation)))

# use elastic net to find best fit
alphas = np.logspace(-1, 2, 200)
l1_ratios = [0, 0.1, 0.5, 0.95, 0.99, 1]

# Set up Repeated CV: 5 folds, repeated 10 times = 50 total splits
n_splits = 10
n_repeats = 10
total_folds = n_splits * n_repeats
rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=_SEED)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = make_pipeline(
        StandardScaler(),
        # Pass the 'rkf' object to cv here
        ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=rkf, n_jobs=-1)
    )
    model.fit(X_tr, y_tr)
    
elastic = model.named_steps["elasticnetcv"]

training_fit = model.predict(X_tr)
residuals = y_tr - training_fit

# 1 SE rule calculations
l1_ratio_opt = elastic.l1_ratio_
l1_ratio_opt_id = np.argwhere(np.abs(np.array(l1_ratios) - l1_ratio_opt) < 10e-5)[0][0]
coefs_opt = elastic.coef_

# mses shape is now (n_alphas, total_folds)
mses = elastic.mse_path_[l1_ratio_opt_id] 

# Calculate the RMSE per fold FIRST, then take the mean and standard deviation
rmses = np.sqrt(mses)
mean_rmse = np.mean(rmses, axis=1) # The mean of the validation RMSEs
SEs = np.std(rmses, axis=1) / np.sqrt(total_folds) # Standard Error across all 50 folds

min_rmse_id = np.argmin(mean_rmse)
min_rmse = mean_rmse[min_rmse_id]
alpha_opt = alphas[min_rmse_id]

# Find the 1-SE alpha
threshold = min_rmse + SEs[min_rmse_id]
one_se_id = np.max(np.argwhere(mean_rmse < threshold))
max_alpha = alphas[one_se_id]
estimated_test_rmse = mean_rmse[one_se_id]

print(f'Non-zero coefficients (CV model): {len(np.nonzero(coefs_opt)[0])}')

# Fit the FINAL model using the standard ElasticNet class (no CV needed here)
final_model = make_pipeline(
    StandardScaler(),
    ElasticNet(alpha=max_alpha, l1_ratio=l1_ratio_opt) 
)
final_model.fit(X_tr, y_tr)

# Predict on test data
y_pred_tst = final_model.predict(X_tst)

np.savetxt("data/predictions.csv", y_pred_tst, delimiter=",")

elasticmodel = final_model.named_steps["elasticnet"]
coefs_opt_tst = elasticmodel.coef_

# Plotting
plt.figure(figsize=(8, 5))
plt.errorbar(alphas, mean_rmse, yerr=SEs, color='navy', ecolor='lightsteelblue', label='Standard Errors')
plt.axvline(max_alpha, color='green', linestyle='-.', alpha=0.4, label='1SE $\\lambda$')
plt.axhline(threshold, color='red', linestyle='-.', alpha=0.4, label='Threshold')
plt.axvline(alpha_opt, color='blue', linestyle='-.', alpha=0.4, label='Minimum RMSE')

plt.legend(loc='best')
plt.xscale('log') # plt.semilogx() works, but setting scale is often cleaner with errorbars
plt.xlabel('Alpha ($\\lambda$)')
plt.ylabel('Mean CV RMSE')
plt.tight_layout()
plt.grid()

plt.show()

print(f'Optimal l1_ratio: {l1_ratio_opt}\n'
      f'Optimal lambda: {alpha_opt:.4f}\n'
      f'1SE max lambda: {max_alpha:.4f}\n'
      f'Min RMSE = {min_rmse:.4f}\n'
      f'1SE RMSE Estimate = {estimated_test_rmse:.4f}'
      )

# 'mses' currently holds the errors for the optimal l1_ratio across all alphas and folds.
fold_mses_1se = mses[one_se_id]
fold_rmses_1se = np.sqrt(fold_mses_1se)

mean_rmse_1se = np.mean(fold_rmses_1se)
n_folds = len(fold_rmses_1se) # 50 (5 splits * 10 repeats)
std_rmse_1se = np.std(fold_rmses_1se)
se_rmse_1se = std_rmse_1se / np.sqrt(n_folds)

# 4. Calculate the 95% Confidence Interval using the t-distribution
confidence_level = 0.95
degrees_freedom = n_folds - 1

ci_lower, ci_upper = stats.t.interval(
    confidence=confidence_level, 
    df=degrees_freedom, 
    loc=mean_rmse_1se, 
    scale=std_rmse_1se
)

kliep = DensityRatioEstimator(sigmas=[max_alpha])
kliep.fit(X_tr, X_tst)  # keyword arguments are X_train and X_test
w = kliep.predict(X_tr)
w_norm = w / w.sum()
rmse_estimate = np.sqrt(np.sum(w_norm * residuals**2))
print(rmse_estimate)

print("--- 95% Confidence Interval for 1-SE RMSE ---")
print(f"Mean RMSE Estimate: {mean_rmse_1se:.4f}")
print(f"KLIEP estimate: {rmse_estimate:.4f}")
print(f"Standard Error:     {se_rmse_1se:.4f}")
print(f"95% CI:             [{ci_lower:.4f}, {ci_upper:.4f}]")

# Optional: Visualize the distribution of the fold errors
plt.figure(figsize=(7, 4))
sns.histplot(fold_rmses_1se, kde=True, bins=14, color='steelblue', alpha=0.6)
plt.axvline(mean_rmse_1se, color='red', linestyle='--', label=f'Mean: {mean_rmse_1se:.4f}')
plt.axvline(ci_lower, color='green', linestyle=':', label=f'95% CI Lower: {ci_lower:.4f}')
plt.axvline(ci_upper, color='green', linestyle=':', label=f'95% CI Upper: {ci_upper:.4f}')

plt.title(f'Distribution of Validation RMSEs ({total_folds} Repeated Folds)')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

