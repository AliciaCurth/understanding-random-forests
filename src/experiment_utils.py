import os
import csv

import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from effective_parameters import get_bootstrap_weights, create_S_from_tree
from data_utils import marsadd_dgp, marsmult_dgp, generate_mars_data
from metric_utils import track_metrics_through_single_forest, track_metrics_through_single_gbtreg

def create_file_and_writer(file_name, res_dir, header):
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if not os.path.isfile(res_dir + (file_name + ".csv")):
        # if file does not exist yet
        out_file = open(res_dir + (file_name + ".csv"), "w", newline='', buffering=1)
        writer = csv.writer(out_file)
        writer.writerow(header)
    else:
        # just open existing file
        out_file = open(res_dir + (file_name + ".csv"), "a", newline='', buffering=1)
        writer = csv.writer(out_file)
  
    return out_file, writer


HEADER_BOOSTING =  ['max_leaf_nodes', 'learning_rate', 'seed', 
                             'dataset', 'sigma', 'd',
          'n_trees_tot', 'n_trees_curr',
            'mse_train', 'mse_test',
            'acc_train', 'acc_test',
            'eff_p_tr',
           'l2_train_sq', 'l2_test_sq',
           'mse_train_resamp', 'mse_train_true']


HEADER_RF =  ['max_leaf_nodes', 'max_features', 'bootstrap',
               'seed', 'dataset', 'sigma', 'd',
          'n_trees_tot', 'n_trees_curr',
            'mse_train', 'mse_test',
            'acc_train', 'acc_test',
            'eff_p_tr',
           'l2_train_sq', 'l2_test_sq',
           'mse_train_resamp', 'mse_train_true']


GB_PARAMS = {'max_features': None,
             'max_depth': None,
             'init': 'zero',
             'criterion': 'squared_error',
             'min_samples_leaf': 1,
             'min_samples_split': 2,
             }

RF_PARAMS = {
            'max_depth': None,
             }


def run_simulated_experiments(file_name, configs, n_reps=10, boosting=False, d=5, n_train=500, n_test=500, marsadd=True, sigma=1,
                                 save_file=True, res_dir='results/', verbose=True):
   random_states = np.arange(1, n_reps+1)

   if save_file:
      out_file, writer = create_file_and_writer(file_name + "_boost" if boosting else file_name, res_dir, HEADER_BOOSTING if boosting else HEADER_RF)
   
   out_frame = pd.DataFrame(columns=HEADER_BOOSTING if boosting else HEADER_RF)
   
   for base_seed in random_states:
       # generate data
       X_train, y_train, X_test, y_test = generate_mars_data(n_train=n_train, n_test=n_test, d=d, sigma=sigma, seed=base_seed, marsadd=marsadd)
       y_train_resamp = marsadd_dgp(X_train, sigma=sigma) if marsadd else marsmult_dgp(X_train, sigma=sigma)
       y_train_true = marsadd_dgp(X_train, sigma=0) if marsadd else marsmult_dgp(X_train, sigma=0)

       for idx, config in enumerate(configs):
        if verbose:
             if boosting:
                print("Running boosting experiment with seed {}, n_estimators {}, learning rate {} and max_leaf_nodes {}.".format(base_seed,
                                                                                             config['n_estimators'],
                                                                                              config['learning_rate'],
                                                                                             config['max_leaf_nodes'] 
                                                                                            ))
             else:
                print("Running forest experiment with seed {}, n_estimators {}, max features {}, bootstrap {} and max_leaf_nodes {}.".format(base_seed,
                                                                                             config['n_estimators'],
                                                                                             config['max_features'],
                                                                                             config['bootstrap'],
                                                                                             config['max_leaf_nodes'], 
                                                                                             ))
        if boosting: 
            clf = GradientBoostingRegressor(n_estimators=config['n_estimators'],
                                        max_leaf_nodes=config['max_leaf_nodes'],
                                        learning_rate=config['learning_rate'],
                                    random_state = base_seed, **GB_PARAMS)
        else:
            clf = RandomForestRegressor(n_estimators=config['n_estimators'],
                                    max_leaf_nodes=config['max_leaf_nodes'],
                                    bootstrap=config['bootstrap'],
                                    max_features=config['max_features'],
                                    random_state = base_seed, **RF_PARAMS)

        clf.fit(X_train, y_train)
        if boosting:
            out_iter = track_metrics_through_single_gbtreg(clf, X_train, X_test,
                                                       y_train,
                                                       y_test,
                                                       verbose>1, y_train_resamp, y_train_true)
            next_frame = pd.concat([pd.DataFrame(columns=['max_leaf_nodes', 'learning_rate', 'seed', 'dataset', 'sigma', 'd'], 
                                                 data=[[config['max_leaf_nodes'], config['learning_rate'], base_seed, marsadd, sigma, d]]
                                             ), out_iter], axis=1)
        else:
            out_iter = track_metrics_through_single_forest(clf, X_train, X_test, y_train, y_test,
                                                          verbose>1, y_train_resamp, y_train_true)
            next_frame = pd.concat([pd.DataFrame(columns=['max_leaf_nodes', 'max_features', 'bootstrap', 'seed', 'dataset', 'sigma', 'd'], 
                                               data=[[config['max_leaf_nodes'], config['max_features'], config['bootstrap'], base_seed,  marsadd, sigma, d]]
                                             ), out_iter], axis=1)
        
          

        # write to file
        if save_file:
            for i in range(next_frame.shape[0]):
                next_row = next_frame.iloc[i, :].values
                writer.writerow(next_row)

        # update dataframe
        out_frame = pd.concat([out_frame, next_frame])
        
   if save_file:
        out_file.close()
   return out_frame
          
                        

def run_resampling_experiment(forest, dgp, X_train, X_test, n_reps,
                              shuffle_y: bool = False, fix_forest: bool = False,
                              seed=42):
  np.random.seed(42)

  n_train = X_train.shape[0]
  n_test = X_test.shape[0]
  n_trees = forest.n_estimators

  # things to track: true vals, pred train, pred test, errors, EPs
  y_trains = np.zeros(shape=(n_train, n_trees, n_reps))
  y_train_preds = np.zeros(shape=(n_train, n_trees, n_reps))
  y_test_preds = np.zeros(shape=(n_test, n_trees, n_reps))

  mse_trains = np.zeros(shape=(n_trees, n_reps))
  mse_train_resamps = np.zeros(shape=(n_trees, n_reps))
  mse_tests = np.zeros(shape=(n_trees, n_reps))

  ep_trains = np.zeros(shape=(n_trees, n_reps))
  ep_tests = np.zeros(shape=(n_trees, n_reps))

  for i in range(n_reps):
    # resample outcomes
    y_train_i = dgp(X_train)
    y_train_resamp_i = dgp(X_train)
    y_test_i = dgp(X_test)

    if (i == 0) or (not fix_forest):
      # fit new forest
      forest_i = clone(forest)
      if not shuffle_y:
        forest_i.fit(X_train, y_train_i)
      else:
        # shuffle the ys to create random random forest
        y_train_shuffled = y_train_i.copy()
        np.random.shuffle(y_train_shuffled)
        forest_i.fit(X_train, y_train_shuffled)

    # loop through forest to get predictions & metrics
    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))
    for n in range(n_trees):
        if forest.bootstrap:
          bootstrap_weights = get_bootstrap_weights(forest_i.estimators_[n].random_state, n_train, forest_i.max_samples)
        else:
          bootstrap_weights = None
        S_train, S_test = create_S_from_tree(forest_i.estimators_[n], X_train, X_test, bootstrap_weights=bootstrap_weights)

        S_train_curr += S_train
        S_test_curr += S_test

        S_train_i = S_train_curr / S_train_curr.sum(axis=1)[:, None]
        S_test_i = S_test_curr / S_test_curr.sum(axis=1)[:, None]

        y_pred_train_i_n = np.matmul(S_train_i, y_train_i)
        y_pred_test_i_n = np.matmul(S_test_i, y_train_i)

        # save predictions
        y_trains[:, n, i] = y_train_i
        y_train_preds[:, n, i] = y_pred_train_i_n
        y_test_preds[:, n, i] = y_pred_test_i_n

        # compute other metrics
        mse_train, mse_test, acc_train, acc_test, \
        eff_p_tr, l2_norm_train_sq, l2_norm_test_sq, mse_train_resamp, mse_train_true = compute_metrics_from_S(
            S_train_i, S_test_i, y_train_i, y_test_i, y_train_resamp=y_train_resamp_i,
            y_train_true=None)

        # store metrics
        mse_trains[n, i] = mse_train
        mse_tests[n, i] = mse_test
        mse_train_resamps[n, i] = mse_train_resamp
        ep_trains[n, i] = l2_norm_train_sq
        ep_tests[n, i] = l2_norm_test_sq

  # Aggregate to compute:
  # 1) avg of all metrics across reps
  mse_train = np.average(mse_trains, axis=1)
  mse_test = np.average(mse_tests, axis=1)
  mse_train_resamp = np.average(mse_train_resamps, axis=1)
  ep_train = np.average(ep_trains, axis=1)
  ep_test = np.average(ep_tests, axis=1)

  # 2) avg variance of predictions
  var_train_pred = np.average(np.var(y_train_preds, axis=2), axis=0)
  var_test_pred = np.average(np.var(y_test_preds, axis=2), axis=0)

  # 3) sum covariance of train-predictions and train-outcomes (d.o.f. a la mentch & zhou)
  # loop bc np.cov does not seem to have axis arg
  train_pred_cov = np.zeros(shape=(n_trees, ))
  for n in range(n_trees):
    for i in range(n_train):
      train_pred_cov[n] += np.cov(y_trains[i, n, :], y_train_preds[i, n, :])[0, 1]


  return mse_train, mse_test, mse_train_resamp, ep_train, ep_test, var_train_pred, var_test_pred, train_pred_cov


          



