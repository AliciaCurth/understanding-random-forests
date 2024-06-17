import numpy as np

import openml

# utils for simulated data ------------------------
def generate_mars_data(n_train, n_test, d, sigma=1, marsadd=True, seed=42):
  np.random.seed(seed)

  assert d >= 5
  n_total = n_train + n_test

  # generate covariates from uniform model
  X_all = np.random.uniform(low=0, high=1, size=(n_total, d))

  if marsadd:
    y_all = marsadd_dgp(X_all, sigma=sigma)
  else:
    y_all = marsmult_dgp(X_all, sigma=sigma)

  return X_all[:n_train, :], y_all[:n_train], X_all[n_train:, :], y_all[n_train:]


def generate_test_x_from_train_with_offset(X, offset):
  n, d = X.shape
  X_offset = np.random.uniform(low=-offset, high=offset, size=(n, d))

  return X + X_offset


def marsadd_dgp(X, sigma=1):
  n, d = X.shape
  assert d >= 5

  epsilon = np.random.normal(size=n, loc=0, scale=sigma)

  return 0.1 * np.exp(4 * X[:, 0]) + 4/(1+np.exp(-20*(X[:, 1] -.5))) + 3*X[:, 2] + 2*X[:, 3] + X[:, 4] + epsilon


def marsmult_dgp(X, sigma=1):
  n, d = X.shape
  assert d >= 5

  epsilon = np.random.normal(size=n, loc=0, scale=sigma)
  return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - .05)**2 + 10*X[:, 3] + 5 * X[:, 4] + epsilon


class MARS_DGP(object):
  def __init__(self, marsadd: bool = True, sigma: float = 1):
    self.marsadd = marsadd
    self.sigma = sigma

  def __call__(self, X):
    if self.marsadd:
      return marsadd_dgp(X, self.sigma)
    else:
      return marsmult_dgp(X, self.sigma)



# utils for openml data -----------------------------
def get_openml_data(id, classification: bool = False):
  dataset = openml.datasets.get_dataset(id)
  X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
    )
  if classification:
    # center for convenience
    y = y.astype(float) - .5

  return X.values, y.values

def subsample_and_split_data(X, y, n_train, n_test, seed):
  # subsample data
  np.random.seed(seed)

  n_sel = n_train + n_test
  n_total, d = X.shape

  idx_sel = np.random.choice(np.arange(n_total), size=n_sel, replace=False)

  return X[idx_sel[:n_train], :], y[idx_sel[:n_train]], X[idx_sel[n_train:], :], y[idx_sel[n_train:]]
