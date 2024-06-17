import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from src.effective_parameters import get_bootstrap_weights, create_S_from_tree, create_S_from_single_boosted_tree

def compute_metrics_from_S(S_train, S_test, y_train, y_test,
                           y_train_resamp=None, y_train_true=None):
    # compute predictions
    y_pred_train = np.matmul(S_train, y_train)
    y_pred_test = np.matmul(S_test, y_train)

    # compute MSE and missclassification
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    if y_train_resamp is not None:
      mse_train_resamp = mean_squared_error(y_train_resamp, y_pred_train)
    else:
      mse_train_resamp = None

    if y_train_true is not None:
      mse_train_true = mean_squared_error(y_train_true, y_pred_train)
    else:
      mse_train_true = None

    acc_train = 1 - np.mean((y_pred_train > 0) == (y_train > 0))
    acc_test = 1 - np.mean((y_pred_test > 0) == (y_test > 0))

    # compute trace metric
    eff_p_tr = np.trace(S_train)

    # compute l2-norm
    l2_norm_train_sq = np.mean(np.linalg.norm(S_train, axis=1) ** 2)
    l2_norm_test_sq = np.mean(np.linalg.norm(S_test, axis=1) ** 2)

    return mse_train, mse_test, acc_train, acc_test, eff_p_tr, l2_norm_train_sq, l2_norm_test_sq, mse_train_resamp, mse_train_true


def track_metrics_through_single_forest(forest, X_train, X_test,
                                        y_train, y_test,
                                        verbose=0, y_train_resamp=None,
                                        y_train_true=None):
    # track change in metrics for single forest as we loop through estimators

    header = ['n_trees_tot',  'n_estimators',
                  'train_error_mse', 'test_error_mse',
                  'train_error_binary', 'test_error_binary',
              'eff_p_tr',
              'l2_train_sq', 'l2_test_sq',
              'mse_train_resamp', 'mse_train_true']

    # create frame to save results in
    out_frame = pd.DataFrame(columns=header)

    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(forest.estimators_)

    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))

    for n in range(n_trees):
        if verbose > 0:
            print('Computing S for forest with {} trees.'.format(n + 1))

        if forest.bootstrap:
          bootstrap_weights = get_bootstrap_weights(forest.estimators_[n].random_state, n_train, forest.max_samples)
        else:
          bootstrap_weights = None

        S_train, S_test = create_S_from_tree(forest.estimators_[n], X_train, X_test, bootstrap_weights=bootstrap_weights)
        S_train_curr += S_train
        S_test_curr += S_test

        if verbose > 0:
            print('Computing metrics for forest with {} trees.'.format(n + 1))

        mse_train, mse_test, acc_train, acc_test, \
        eff_p_tr, l2_norm_train_sq, l2_norm_test_sq, mse_train_resamp, mse_train_true = compute_metrics_from_S(
            S_train_curr / S_train_curr.sum(axis=1)[:, None],
            S_test_curr / S_test_curr.sum(axis=1)[:, None],
            y_train, y_test, y_train_resamp=y_train_resamp,
            y_train_true=y_train_true)

        # append data
        next_row = [n_trees, n + 1, mse_train, mse_test,
                    acc_train, acc_test, eff_p_tr,
                    l2_norm_train_sq, l2_norm_test_sq,
                    mse_train_resamp, mse_train_true]

        new_frame = pd.DataFrame(columns=header,
                                 data=[next_row])
        out_frame = pd.concat([out_frame, new_frame])

    return out_frame


def track_metrics_through_single_gbtreg(gbtreg, X_train, X_test, y_train, y_test,
                                        verbose=0, y_train_resamp=None,
                                        y_train_true=None):
    # track change in metrics for single GBReg
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(gbtreg.estimators_)
    lr = gbtreg.learning_rate
    seed = gbtreg.random_state

    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))

    # create frame to save
    header = ['n_boost_tot', 'n_boost',
              'train_error_mse', 'test_error_mse',
              'train_error_binary', 'test_error_binary',
              'eff_p_tr',
              'l2_train_sq', 'l2_test_sq',
               'mse_train_resamp', 'mse_train_true']

    out_frame = pd.DataFrame(columns=header)

    for n in range(n_trees):
        if verbose > 0:
          print('Computing S for tree {}.'.format(n))
        S_train, S_test = create_S_from_single_boosted_tree(gbtreg.estimators_[n][0],
                                                                        None if n == 0 else S_train_curr,
                                                                        X_train, X_test)
        S_train_curr += lr * S_train
        S_test_curr += lr * S_test

        if verbose > 0:
            print('Computing metrics for tree {}.'.format(n))

        mse_train, mse_test, acc_train, acc_test, \
        eff_p_tr,  \
        l2_norm_train_sq, l2_norm_test_sq, mse_train_resamp, mse_train_true = compute_metrics_from_S(
            S_train_curr,
            S_test_curr,
            y_train, y_test, y_train_resamp, y_train_true)

        # append data
        next_row = [n_trees, n + 1, mse_train, mse_test,
                    acc_train, acc_test, eff_p_tr,
                    l2_norm_train_sq, l2_norm_test_sq,
                    mse_train_resamp, mse_train_true]

        new_frame = pd.DataFrame(columns=header,
                                 data=[next_row])
        out_frame = pd.concat([out_frame, new_frame])

    return out_frame


