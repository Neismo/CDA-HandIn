import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LassoCV, RidgeCV,ElasticNetCV
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split
import matplotlib.pyplot as plt
from data import normalize
import seaborn as sns
sns.set_theme() # Set searborn as default


def cross_validate_lasso_alphas(X, y, alphas, n_splits=5, n_repeats=10, random_state=42):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    alpha_rmse_stats = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=100000)
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=rkf,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        rmse_scores = -cv_results['test_score']
        alpha_rmse_stats.append({
            'Alpha': alpha,
            'Mean RMSE': np.mean(rmse_scores),
            'Std RMSE': np.std(rmse_scores)
        })

    return pd.DataFrame(alpha_rmse_stats).sort_values('Alpha').reset_index(drop=True)

def evaluate_models(X, y):
    rkf = RepeatedKFold(n_splits=5, n_repeats=50, random_state=42)
    
    # Define models with internal CV
    models = {
        "Lasso": LassoCV(cv=5, max_iter=10000),
        "Ridge": RidgeCV(cv=5),
        "Elastic Net": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, 1], cv=5, max_iter=10000)
    }
    
    summary_stats = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        cv_results = cross_validate(
            model, X, y,
            cv=rkf, 
            scoring='neg_root_mean_squared_error',
            return_estimator=True,
            n_jobs=-1
        )

        alphas = [est.alpha_ for est in cv_results['estimator']]
        
        summary_stats.append({
            "Model": name,
            "Mean RMSE": np.mean(-cv_results['test_score']),
            "Avg Best Alpha": np.mean(alphas),
            "Alpha Std Dev": np.std(alphas) # High std dev means the model is unstable
        })

    return pd.DataFrame(summary_stats)

if __name__ == "__main__":
    from data import load_data
    # Assuming Y_raw and X are loaded and pre-processed as you described
    Y_raw, X = load_data(file_path="data/processed_data.csv")
    Y = Y_raw.ravel()

    X[:, :95] = normalize(X[:, :95], list(range(95)))[0]
    # X = normalize(X, list(range(95)))[0]

    """
    print(f"Evaluating models on shape: {X.shape}...")
    summary = evaluate_models(X, Y)
    
    print("\n--- Cross-Validation Results (Repeated K-Fold) ---")
    print(summary.to_string(index=False))
    """

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123) # Just to set the seed for permutation importance later
    X_train_standardized, _, _ = normalize(X_train, list(range(95)))
    selected_alphas = np.logspace(-2, 1, 50)
    lasso_alpha_summary = cross_validate_lasso_alphas(X_train_standardized, y_train, selected_alphas)
    print("\n--- Lasso RMSE by Selected Alpha Values ---")
    print(lasso_alpha_summary.to_string(index=False))

    # Find lowest RMSE and corresponding alpha
    best_alpha_idx = lasso_alpha_summary['Mean RMSE'].idxmin()
    best_row = lasso_alpha_summary.loc[best_alpha_idx]
    print(f"\nBest Alpha: {best_row['Alpha']:.4f} with Mean RMSE: {best_row['Mean RMSE']:.4f}")

    # Find lowest RMSE withing the 1-std range of the best alpha
    J = np.where(lasso_alpha_summary['Mean RMSE'] < best_row['Mean RMSE'] + best_row['Std RMSE'])[0]
    
    plt.figure(figsize=(9, 5))
    plt.errorbar(
        lasso_alpha_summary['Alpha'],
        lasso_alpha_summary['Mean RMSE'],
        yerr=lasso_alpha_summary['Std RMSE'],
        fmt='o-',
        capsize=4
    )
    plt.axvline(x=best_row['Alpha'], color='r', linestyle='--', label=f"Best $\\alpha$: {best_row['Alpha']:.4f}")
    plt.axvline(x=lasso_alpha_summary.iloc[J]['Alpha'].iloc[-1], color='g', linestyle='--', label=f"Stable $\\alpha$: {lasso_alpha_summary.iloc[J]['Alpha'].iloc[-1]:.4f}")
    plt.axhline(y=best_row['Mean RMSE'] + best_row['Std RMSE'], color='gray', linestyle='--', label="1-$\\sigma$ Dev Threshold")
    plt.xscale('log')
    plt.xlabel('$\\alpha$ (log scale)')
    plt.ylabel('RMSE')
    plt.title('Lasso Cross-Validation RMSE by $\\alpha$')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Assuming your best model is fitted
    final_lasso = LassoCV(cv=5, max_iter=100000)
    final_lasso.fit(X_train_standardized, y_train)

    # Count how many features are non-zero
    n_features_kept = np.sum(final_lasso.coef_ != 0)
    print(f"Lasso kept {n_features_kept} features and discarded {X.shape[1] - n_features_kept}.")

    # Get predictions
    y_pred = final_lasso.predict(X_test)
    residuals = y_test.ravel() - y_pred

    plt.figure(figsize=(10, 5))

    # Predicted VS actual
    plt.subplot(1, 2, 1)
    plt.scatter(y_test.ravel(), y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs. Actual')

    # Residuals Plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.tight_layout()
    plt.show()

    # 1. Extract coefficients
    # If using a Pipeline, use: coefs = model.named_steps['regressor'].coef_
    coefs = final_lasso.coef_
    feature_names = [f"X_{i}" for i in range(len(coefs))] # Ensure this matches your post-encoding column names

    # 2. Create a DataFrame for easy sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs)
    })

    # 3. Filter for non-zero and take the top 20 by magnitude
    top_20 = importance_df[importance_df['Coefficient'] != 0].nlargest(20, 'Abs_Coefficient')

    # 4. Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_20, 
        x='Coefficient', 
        y='Feature', 
        palette="vlag"
    )
    plt.title('Top 20 Most Influential Features (Lasso)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

    # Permutation importance of features
    result = permutation_importance(
        final_lasso, X, Y,
        n_repeats=10, 
        random_state=42, 
        scoring='neg_root_mean_squared_error'
    )

    # 2. Organize the results
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Mean': result.importances_mean,
        'Importance_Std': result.importances_std
    }).sort_values(by='Importance_Mean', ascending=False)

    # 3. Plot the top 10
    top_10 = importance_df.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_10['Feature'],
        top_10['Importance_Mean'],
        xerr=top_10['Importance_Std']
    )
    plt.gca().invert_yaxis()
    plt.xlabel('Increase in RMSE when shuffled')
    plt.title('Permutation Importance (Top 10 Features)')
    plt.tight_layout()
    plt.show()