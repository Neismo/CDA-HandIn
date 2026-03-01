import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, RidgeCV,ElasticNetCV
from sklearn.model_selection import RepeatedKFold, cross_validate
import matplotlib.pyplot as plt
from data import normalize
import seaborn as sns
sns.set_theme() # Set searborn as default

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

    X = normalize(X, list(range(95)))[0]
    print(X.shape)

    print(f"Evaluating models on shape: {X.shape}...")
    summary = evaluate_models(X, Y)
    
    print("\n--- Cross-Validation Results (Repeated K-Fold) ---")
    print(summary.to_string(index=False))

    # Assuming your best model is fitted
    final_lasso = LassoCV(cv=5, max_iter=10000)
    final_lasso.fit(X, Y)

    # Count how many features are non-zero
    n_features_kept = np.sum(final_lasso.coef_ != 0)
    print(f"Lasso kept {n_features_kept} features and discarded {X.shape[1] - n_features_kept}.")

    # Get predictions
    y_pred = final_lasso.predict(X)
    residuals = Y_raw.ravel() - y_pred

    plt.figure(figsize=(10, 5))

    # Predicted VS actual
    plt.subplot(1, 2, 1)
    plt.scatter(Y_raw.ravel(), y_pred, alpha=0.6)
    plt.plot([Y_raw.min(), Y_raw.max()], [Y_raw.min(), Y_raw.max()], 'r--', lw=2)
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