import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV,ElasticNetCV, Lasso
from sklearn.model_selection import RepeatedKFold, cross_validate
import matplotlib.pyplot as plt
from data import normalize
import seaborn as sns
sns.set_theme() # Set searborn as default

def evaluate_models(X, y):
    # Define the cross-validation strategy: 5 folds, repeated 10 times with different seeds
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    models: dict[str, LinearRegression | LassoCV | RidgeCV | ElasticNetCV] = {
        "OLS (Baseline)": LinearRegression(),
        "Lasso (L1)": LassoCV(cv=5, max_iter=10000),
        "Ridge (L2)": RidgeCV(cv=5),
        "Elastic Net": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, max_iter=10000)
    }
    
    results = []

    for name, model in models.items():
        # cv_results returns scores for every fold in every repetition
        cv_results = cross_validate(
            model, X, y,
            cv=rkf, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            return_estimator=True
        )
        
        # Convert negative RMSE to positive
        rmse_scores = -cv_results['test_score']
        fitted_model = cv_results['estimator'][0]  # Get the first fitted model for hyperparameter info
        
        results.append({
            "Model": name,
            "Mean RMSE": np.mean(rmse_scores),
            "Std RMSE": np.std(rmse_scores)
        })

        if isinstance(fitted_model, (LassoCV, RidgeCV, ElasticNetCV)):
            print(f"{name} - Best alpha: {fitted_model.alpha_}")
            if isinstance(fitted_model, ElasticNetCV):
                print(f"{name} - Best l1_ratio: {fitted_model.l1_ratio_}")
    
    return pd.DataFrame(results)

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
    best_lasso = Lasso(alpha=0.6413483150282301)
    best_lasso.fit(X, Y_raw)

    # Count how many features are non-zero
    n_features_kept = np.sum(best_lasso.coef_ != 0)
    print(f"Lasso kept {n_features_kept} features and discarded {X.shape[1] - n_features_kept}.")

    # Get predictions
    y_pred = best_lasso.predict(X)
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
    coefs = best_lasso.coef_
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
        best_lasso, X, Y,
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