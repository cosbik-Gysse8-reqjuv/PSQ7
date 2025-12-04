import csv
import numpy as np


def load_csv_as_xy(path, y_column_name, x_column_names, add_intercept=True):
    """
    Load a CSV file and return (X, y) numpy arrays for regression.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    y_column_name : str
        Column name for the dependent variable.
    x_column_names : list of str
        List of column names for independent variables.
    add_intercept : bool
        If True, adds a column of ones to X for the intercept.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples, )
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


def ols(X, y):
    """
    Ordinary Least Squares (OLS) regression implemented from scratch.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix (including intercept if desired).
    y : ndarray, shape (n_samples,)
        Dependent variable.

    Returns
    -------
    results : dict
        Contains coefficients, standard errors, t-stats, etc.
    """
    # Ensure correct shapes
    y = y.reshape(-1, 1)  # column vector
    n, k = X.shape

    # Beta_hat = (X'X)^(-1) X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    beta_hat = XtX_inv @ Xty  # (k x 1)

    # Fitted values and residuals
    y_hat = X @ beta_hat
    residuals = y - y_hat

    # Sum of squared residuals (SSR) and total sum of squares (SST)
    SSR = float(residuals.T @ residuals)
    y_mean = float(y.mean())
    SST = float(((y - y_mean)**2).sum())

    # Unbiased estimate of variance of the error term
    sigma2_hat = SSR / (n - k)

    # Variance-covariance matrix of beta_hat
    var_beta_hat = sigma2_hat * XtX_inv
    se_beta_hat = np.sqrt(np.diag(var_beta_hat)).reshape(-1, 1)

    # t-statistics for each coefficient
    t_stats = beta_hat / se_beta_hat

    # R-squared and adjusted R-squared
    r_squared = 1 - SSR / SST
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)

    return {
        "beta_hat": beta_hat.flatten(),       # coefficients
        "se_beta_hat": se_beta_hat.flatten(), # standard errors
        "t_stats": t_stats.flatten(),         # t-statistics
        "y_hat": y_hat.flatten(),             # fitted values
        "residuals": residuals.flatten(),     # residuals
        "sigma2_hat": sigma2_hat,
        "SSR": SSR,
        "SST": SST,
        "R2": r_squared,
        "adj_R2": adj_r_squared,
        "n": n,
        "k": k,
    }


def print_ols_summary(results, x_column_names, add_intercept=True):
    """
    Pretty-print a simple OLS summary.

    Parameters
    ----------
    results : dict
        Output of the ols() function.
    x_column_names : list of str
        Names of the independent variables (without intercept).
    add_intercept : bool
        If True, assumes the first coefficient is the intercept.
    """
    beta = results["beta_hat"]
    se = results["se_beta_hat"]
    t = results["t_stats"]

    # Build names list
    names = []
    if add_intercept:
        names.append("Intercept")
    names.extend(x_column_names)

    print("==============================================")
    print("               OLS Regression                 ")
    print("==============================================")
    print(f"Number of observations: {results['n']}")
    print(f"Number of parameters:   {results['k']}")
    print("----------------------------------------------")
    print(f"R-squared:      {results['R2']:.4f}")
    print(f"Adj. R-squared: {results['adj_R2']:.4f}")
    print(f"Sigma^2 (s^2):  {results['sigma2_hat']:.4f}")
    print("==============================================")
    print("{:<15s} {:>12s} {:>12s} {:>12s}".format("Variable", "Coef", "Std.Err", "t-Stat"))
    print("----------------------------------------------")
    for name, b, s, tt in zip(names, beta, se, t):
        print("{:<15s} {:>12.4f} {:>12.4f} {:>12.4f}".format(name, b, s, tt))
    print("==============================================")


if __name__ == "__main__":
    # Example usage:
    #
    # Suppose you have a CSV file "data.csv" like:
    # y,x1,x2
    # 5.1,1.0,3.2
    # 4.9,2.0,3.1
    # ...
    #
    # And you want to regress y on x1 and x2.

    csv_path = "data.csv"
    y_col = "y"
    x_cols = ["x1", "x2"]

    X, y = load_csv_as_xy(csv_path, y_col, x_cols, add_intercept=True)
    results = ols(X, y)
    print_ols_summary(results, x_cols, add_intercept=True)
