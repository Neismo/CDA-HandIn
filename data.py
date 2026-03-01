import numpy as np
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer
sns.set_theme() # Set searborn as default

def load_data(file_path: str = "data/case1Data.csv") -> tuple[np.ndarray, np.ndarray]:
    # Load in the CSV data using numpy. The first column is the Y values.
    # No processing is done here.
    YX = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    Y, X = YX[:, 0], YX[:, 1:]  
    return Y, X

def preprocess_features(
    X: np.ndarray,
    drop_feature_1based: int = 96,
    discrete_unique_max: int = 10,
    discrete_ratio_max: float = 0.05,
    integer_tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    n_samples, n_features = X.shape
    drop_idx = drop_feature_1based - 1

    if not (0 <= drop_idx < n_features):
        raise ValueError(
            f"Requested feature {drop_feature_1based} to drop, but X has only {n_features} features."
        )

    keep_mask = np.ones(n_features, dtype=bool)
    keep_mask[drop_idx] = False
    X_reduced = X[:, keep_mask]
    remaining_original_indices = np.arange(n_features)[keep_mask]

    categorical_mask = np.zeros(X_reduced.shape[1], dtype=bool)
    for idx in range(X_reduced.shape[1]):
        values = X_reduced[:, idx]
        values = values[~np.isnan(values)]

        if len(values) == 0:
            continue

        n_unique = np.unique(values).size
        unique_ratio = n_unique / len(values)
        is_integer_like = np.all(np.abs(values - np.round(values)) < integer_tol)

        categorical_mask[idx] = is_integer_like and (
            n_unique <= discrete_unique_max or unique_ratio <= discrete_ratio_max
        )

    cat_idx = np.where(categorical_mask)[0]
    cont_idx = np.where(~categorical_mask)[0]

    if len(cont_idx) > 0:
        cont_imputer = SimpleImputer(
            strategy="mean",
            missing_values=np.nan,
        )
        X_cont = cont_imputer.fit_transform(X_reduced[:, cont_idx])
    else:
        cont_imputer = None
        X_cont = np.empty((n_samples, 0))

    if len(cat_idx) > 0:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
            dtype=np.float64,
        )

        raw_cat = X_reduced[:, cat_idx]
        rounded_cat = np.round(raw_cat)
        cat_data = np.empty(raw_cat.shape, dtype=object)
        nan_mask = np.isnan(raw_cat)
        cat_data[nan_mask] = "__nan__"
        cat_data[~nan_mask] = rounded_cat[~nan_mask].astype(np.int64).astype(str)

        X_cat = encoder.fit_transform(cat_data)
    else:
        encoder = None
        X_cat = np.empty((n_samples, 0))

    X_processed = np.hstack([X_cont, X_cat])

    info = {
        "dropped_feature_1based": drop_feature_1based,
        "remaining_feature_count": int(X_reduced.shape[1]),
        "categorical_features_1based": (remaining_original_indices[cat_idx] + 1).tolist(),
        "continuous_features_1based": (remaining_original_indices[cont_idx] + 1).tolist(),
        "processed_feature_count": int(X_processed.shape[1]),
        "encoder": encoder,
        "continuous_imputer": cont_imputer,
    }
    return X_processed, info

def save_data(Y: np.ndarray, X: np.ndarray, file_path: str = "data/processed_data.csv") -> None:
    # Save the processed data to a new CSV file. The first column should be Y, followed by the features in X.
    data_to_save = np.hstack([Y.reshape(-1, 1), X])
    header = ",".join(["Y"] + [f"X{i+1}" for i in range(X.shape[1])])
    np.savetxt(file_path, data_to_save, delimiter=",", header=header, comments="")

def centerData(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    mu = np.mean(data,axis=0)
    data = data - mu
    
    return data, mu

def normalize(data: np.ndarray, feature_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    sub_data = data[:, feature_indices]
    mu = np.mean(sub_data, axis=0)
    sd = np.std(sub_data, axis=0)

    data = data.copy()
    data[:, feature_indices] = (data[:, feature_indices] - mu)/sd
    return data, mu, sd

if __name__ == "__main__":
    Y_raw, X_raw = load_data()
    X, preprocess_info = preprocess_features(X_raw)
    print(
        f"Dropped feature {preprocess_info['dropped_feature_1based']}. "
        f"Categorical features encoded: {len(preprocess_info['categorical_features_1based'])}. "
        f"Total features after preprocessing: {preprocess_info['processed_feature_count']}"
    )
    save_data(Y_raw, X)
