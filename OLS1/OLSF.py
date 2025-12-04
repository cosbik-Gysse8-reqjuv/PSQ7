import csv
import numpy as np


def load_csv_as_xy(path, y_column_name, x_column_names, add_intercept=True):
    """
    Exists to read my CSV file and return (X, y) numpy arrays for regression.
    """
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    y_list = []
    X_list = []

    for row in rows:
        y_list.append(float(row[y_column_name]))
        x_row = []
        for col in x_column_names:
            x_row.append(float(row[col]))
        X_list.append(x_row)

    y = np.array(y_list, dtype=float)
    X = np.array(X_list, dtype=float)

    if add_intercept:
        # Add a column of ones as the first column of X
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack([intercept, X])

    return X, y


import numpy as np

def ols_coefficients(X, y):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)

    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta_hat.flatten()

def homo_cov(X, y, beta_hat):
    """
    Returns the homoskedastic asymptotic covariance matrix:
    sigma^2 * (X'X)^(-1)
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    beta_hat = beta_hat.reshape(-1, 1)

    n, k = X.shape
    residuals = y - X @ beta_hat
    sigma2_hat = float(residuals.T @ residuals) / (n - k)

    XtX_inv = np.linalg.inv(X.T @ X)
    cov_hat = sigma2_hat * XtX_inv
    return cov_hat

def robust_cov(X, y, beta_hat):
    """
    Heteroskedastic-robust covariance matrix:
    (X'X)^(-1) [X' Omegahat X] (X'X)^(-1)
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1, 1)
    beta_hat = beta_hat.reshape(-1, 1)

    XtX_inv = np.linalg.inv(X.T @ X)
    residuals = y - X @ beta_hat              # (n x 1)

    Omega_hat = np.diag((residuals.flatten())**2)  # (n x n)

    # meat = X' Omega_hat X
    meat = X.T @ Omega_hat @ X               # X' diag(e_i^2) X

    cov_hat = XtX_inv @ meat @ XtX_inv
    return cov_hat

csv_path = "OLS1/PS6.csv"
y_col = "Y"
x_cols = ["Treat"]
X, y = load_csv_as_xy(csv_path, y_col, x_cols, add_intercept=True)
beta_hat = ols_coefficients(X, y)
cov_hat = robust_cov(X, y, beta_hat)
print("OLS Coefficients: (Beta0, Beta1)", ols_coefficients(X, y))
print("Robust Variance:", cov_hat)
x_cols = ["Treat", "X1", "X2", "X3"]
X, y = load_csv_as_xy(csv_path, y_col, x_cols, add_intercept=True)
beta_hat = ols_coefficients(X, y)
cov_hat = robust_cov(X, y, beta_hat)
print("OLS Coefficients: (Gamma0, Gamma1, Gammas2-4)", ols_coefficients(X, y))
print("Robust Variance:", cov_hat)