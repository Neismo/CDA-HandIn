import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from time import time


def split_train_test(data, test_size=0.25):
    data.sort_values(by=['Individual'], kind='mergesort', inplace=True)
    X_tr = data[data['Individual'] <= (1-test_size)*data['Individual'].max()]
    X_test = data[data['Individual'] > (1-test_size)*data['Individual'].max()]
    return X_tr, X_test


def replace_nan_with_gaussian(column, output_type: str):
    mean = np.mean(column)
    std = np.std(column)
    nan_id_tr = np.array(np.where(column.isnull()))
    for row_id in nan_id_tr:
        if output_type == 'float':
            column[row_id] = np.random.normal(loc=mean, scale=std)
        elif output_type == 'int':
            column[row_id] = np.round(np.random.normal(loc=mean, scale=std))


def clean_data(X):
    string_cols = X.select_dtypes(include='string').columns.tolist()
    X = X.drop(columns=string_cols, inplace=False)
    nan_id = np.array(np.where(X.isnull()))
    feature_means = np.nanmean(X.iloc[nan_id[1]], axis=0)
    feature_stds = np.nanstd(X.iloc[nan_id[1]], axis=0)
    for nan_element in nan_id.T:
        row_id, col_id = nan_element
        col_type = X.dtypes.iloc[col_id]
        X.iloc[row_id, col_id] = np.array(np.round(np.random.normal(loc=feature_means[col_id], scale=feature_stds[col_id]))).astype(col_type)
    return X


if __name__ == '__main__':
    t = time()
    np.random.seed(42)
    df = pd.read_csv(f'../data/HR_data.csv')
    train, test = split_train_test(df, test_size=0.2)
    train = clean_data(train)
    test = clean_data(test)