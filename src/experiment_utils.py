import os
import csv

import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.metrics import mean_squared_error

from src.effective_parameters import get_bootstrap_weights, create_S_from_tree
from src.data_utils import marsadd_dgp, marsmult_dgp, generate_mars_data, MARS_DGP, generate_test_x_from_train_with_offset, get_openml_data, subsample_and_split_data
from src.metric_utils import track_metrics_through_single_forest, track_metrics_through_single_gbtreg, compute_metrics_from_S


# utils for saving results
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
                             'dataset', 'sigma', 'd', 'offset',
          'n_trees_tot', 'n_estimators',
            'mse_train', 'mse_test',
            'acc_train', 'acc_test',
           'ep_train', 'ep_test',
           'mse_train_resamp', 'mse_train_true']


HEADER_RF =  ['max_leaf_nodes', 'max_features', 'bootstrap',
               'seed', 'dataset', 'sigma', 'd', 'offset',
          'n_trees_tot', 'n_estimators',
            'mse_train', 'mse_test',
            'acc_train', 'acc_test',
           'ep_train', 'ep_test',
           'mse_train_resamp', 'mse_train_true']

HEADER_DECOMP = ['iteration', 'seed', 'max_leaf_nodes', 
                  'max_features', 'bootstrap', 'dataset', 'sigma', 
                  'd', 'offset',
                  'n_trees_tot', 'n_estimators',
                  'mse_train', 'mse_test',
                  'acc_train', 'acc_test',
                  'ep_train', 'ep_test',
                  'mse_train_resamp', 'mse_train_true']

HEADER_DECOMP_SUMMARY =['seed', 'n_estimators',
                         'max_leaf_nodes', 'max_features', 
                         'bootstrap', 'dataset', 'sigma', 'd', 
                         'offset', 
                         'avg_mse', 'repbias', 'modvar', 'avg_excess_mse']


HEADER_RESAMP = ['seed', 'n_reps','shuffled_outcomes',
                 'fixed_forest', 'max_leaf_nodes', 'max_features',
                  'bootstrap', 'dataset', 'sigma', 'd',
                  'n_estimators',
                    'mse_train', 'mse_test', 'mse_train_resamp',
                    'ep_train', 'ep_test',
                    'dof', 'var_train_pred', 'var_test_pred'
                  ]

HEADER_REAL = ['iteration', 'seed', 'max_leaf_nodes', 
                  'max_features', 'bootstrap', 'dataset', 'classification', 
                  'n_train', 'n_test',
                  'n_trees_tot', 'n_estimators',
                  'mse_train', 'mse_test',
                  'acc_train', 'acc_test',
                  'ep_train', 'ep_test',
                  'mse_train_resamp', 'mse_train_true']


HEADER_DECOMP_REAL_SUMMARY =['seed', 'n_estimators',
                         'max_leaf_nodes', 'max_features', 
                         'bootstrap', 'dataset', 'classification', 'n_train', 'n_test', 
                         'avg_mse', 'repbias', 'modvar', 'avg_excess_mse']


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

# SIMULATION EXPERIMENTS ---------------------------------------------------
# code to run experiments in Section 3 (Sec 3.1 and 3.3)----------
def run_simulated_experiments(file_name, configs, n_seeds=10, boosting=False, d=5, n_train=500, n_test=500, marsadd=True, sigma=1,
                                 save_file=True, res_dir='results/', verbose=True, offsets=None):
  if offsets is None:
    offsets = [None]
  
  random_states = np.arange(1, n_seeds+1)

  if save_file:
    out_file, writer = create_file_and_writer(file_name + "_boost" if boosting else file_name, res_dir, HEADER_BOOSTING if boosting else HEADER_RF)
   
  out_frame = pd.DataFrame(columns=HEADER_BOOSTING if boosting else HEADER_RF)
   
  for base_seed in random_states:
      for offset in offsets:
        # generate data
        X_train, y_train, X_test, y_test = generate_mars_data(n_train=n_train, n_test=n_test, d=d, sigma=sigma, seed=base_seed, marsadd=marsadd)
        y_train_resamp = marsadd_dgp(X_train, sigma=sigma) if marsadd else marsmult_dgp(X_train, sigma=sigma)
        y_train_true = marsadd_dgp(X_train, sigma=0) if marsadd else marsmult_dgp(X_train, sigma=0)

        if offset is not None:
          # regenerate data with applicable offset
          X_test = generate_test_x_from_train_with_offset(X_train, offset)
          y_test = marsadd_dgp(X_test, sigma=sigma) if marsadd else marsmult_dgp(X_test, sigma=sigma)

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
              next_frame = pd.concat([pd.DataFrame(columns=['max_leaf_nodes', 'learning_rate', 'seed', 'dataset', 'sigma', 'd', 'offset'], 
                                                    data=[[config['max_leaf_nodes'], config['learning_rate'], base_seed, marsadd, sigma, d, offset]]
                                                ), out_iter], axis=1)
          else:
              out_iter = track_metrics_through_single_forest(clf, X_train, X_test, y_train, y_test,
                                                            verbose>1, y_train_resamp, y_train_true)
              next_frame = pd.concat([pd.DataFrame(columns=['max_leaf_nodes', 'max_features', 'bootstrap', 'seed', 'dataset', 'sigma', 'd', 'offset'], 
                                                  data=[[config['max_leaf_nodes'], config['max_features'], config['bootstrap'], base_seed,  marsadd, sigma, d, offset]]
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
          
                        
# Utils to run experiments in section 3.3 and appendix A ----------
def run_one_resampling_experiment(forest, dgp, X_train, X_test, n_reps,
                              shuffle_y: bool = False, 
                              fix_forest: bool = False,
                              seed=42):
  # optional arguments for variance experiments in appendix:
  # shuffle_y: randomly shuffles labels for fitting forest (mimics a totally randomized tree)
  # fix forest: fixes the forest across resampling of labels (mimics getting rid of randomness in tree building)

  np.random.seed(seed)

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


def run_resampling_experiments(file_name, configs, n_reps=50, n_seeds=10, boosting=False, d=5, n_train=500, n_test=500, marsadd=True, sigma=1,
                                 save_file=True, res_dir='results/', verbose=True, shuffle_y=False, fix_forest=False):
  # optional arguments for variance experiments in appendix:
  # shuffle_y: randomly shuffles labels for fitting forest (mimics a totally randomized tree)
  # fix forest: fixes the forest across resampling of labels (mimics getting rid of randomness in tree building)

  # create file 
  if save_file:
    out_file, writer = create_file_and_writer(file_name, res_dir, HEADER_RESAMP)
   
  out_frame = pd.DataFrame(columns=HEADER_RESAMP)

  random_states = np.arange(1, n_seeds+1)

  for base_seed in random_states:
    # generate data
    X_train, y_train, X_test, y_test = generate_mars_data(n_train=n_train, n_test=n_test, d=d, sigma=sigma, seed=base_seed, marsadd=marsadd)
    dgp = MARS_DGP(marsadd=marsadd, sigma=sigma)

    for idx, config in enumerate(configs):
      # run for different configs
      if verbose:
        print("Running {} experiment with seed {}, bootstrap {}, n_estimators {}, max_features {} and max_leaf_nodes {}.".format(n_reps, base_seed,
                                                                                                             config['bootstrap'],
                                                                                             config['n_estimators'], config['max_features'],
                                                                                             config['max_leaf_nodes']))
      clf = RandomForestRegressor(n_estimators=config['n_estimators'], max_leaf_nodes=config['max_leaf_nodes'],
                                    bootstrap=config['bootstrap'], max_features=config['max_features'],
                                    random_state = base_seed, **RF_PARAMS)

      mse_train, mse_test, mse_train_resamp,ep_train, ep_test, var_train_pred, var_test_pred, train_pred_cov = run_one_resampling_experiment(clf, dgp, X_train, X_test, n_reps, 
                                                                                                                                            shuffle_y=shuffle_y, fix_forest=fix_forest, 
                                                                                                                                            seed=(base_seed+2)*5432)
            

      # create dataframe of results
      res_i = pd.DataFrame(columns=['n_estimators',
                    'mse_train', 'mse_test', 'mse_train_resamp',
                    'ep_train', 'ep_test',
                    'dof', 'var_train_pred', 'var_test_pred'],
                      data=np.transpose(np.array([np.arange(1, config['n_estimators']+1),
                                    mse_train, mse_test, mse_train_resamp,
                                    ep_train, ep_test, train_pred_cov,
                                    var_train_pred, var_test_pred])))

      # append to existing results
      next_frame = pd.concat([pd.concat([pd.DataFrame(columns=['seed', 'n_reps','shuffled_outcomes','fixed_forest', 'max_leaf_nodes', 'max_features', 'bootstrap', 'dataset', 'sigma', 'd'],
                                          data=[[base_seed, n_reps, shuffle_y, fix_forest, config['max_leaf_nodes'], config['max_features'], config['bootstrap'], marsadd, sigma, d]])] * config['n_estimators'], axis=0).reset_index(drop=True),
                                  res_i], axis=1)

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


# can be used to replicate the analysis in Figure 13 in Section 4 ------------
def run_error_decomp_experiments(file_name, configs, offsets=None, n_reps=50, n_seeds=10, save_file=True, res_dir='results/', verbose=True,
                                d=5, n_train=500, n_test=500, marsadd=True, sigma=0):
  out_file, writer = create_file_and_writer(file_name, res_dir, HEADER_DECOMP)

  out_frame = pd.DataFrame(columns=HEADER_DECOMP)
  summary_frame = pd.DataFrame(columns=HEADER_DECOMP_SUMMARY)

  random_states = np.arange(1, n_seeds+1)

  if offsets is None:
    offsets = [None]

  for base_seed in random_states:
    for offset in offsets:
      X_train, y_train, X_test, y_test = generate_mars_data(n_train=n_train, n_test=n_test, d=d, sigma=sigma, seed=base_seed, marsadd=marsadd)
      if offset is not None:
        # regenerate data with applicable offset
        X_test = generate_test_x_from_train_with_offset(X_train, offset)
        y_test = marsadd_dgp(X_test, sigma=sigma) if marsadd else marsmult_dgp(X_test, sigma=sigma)

      y_train_resamp = marsadd_dgp(X_train, sigma=sigma) if marsadd else marsmult_dgp(X_train, sigma=sigma)
      y_train_true = marsadd_dgp(X_train, sigma=0) if marsadd else marsmult_dgp(X_train, sigma=0)

      for idx, config in enumerate(configs):
        predictions = np.zeros(shape=(n_test, config['n_estimators'], n_reps))
        for rep in range(n_reps):
            if verbose:
              print("Running experiment with seed {} version {}, bootstrap {}, n_estimators {}, max_features {} and max_leaf_nodes {}.".format(base_seed, rep,
                                                                                                              config['bootstrap'],
                                                                                              config['n_estimators'],
                                                                                                config['max_features'],
                                                                                              config['max_leaf_nodes']))
            clf = RandomForestRegressor(n_estimators=config['n_estimators'],
                                      max_leaf_nodes=config['max_leaf_nodes'],
                                      bootstrap=config['bootstrap'],
                                      max_features=config['max_features'],
                                      random_state = base_seed*(rep+1), **RF_PARAMS)

            clf.fit(X_train, y_train)

            for n_trees in range(config['n_estimators']):
              # save current predictions
              predictions[:, n_trees, rep] = clf.estimators_[n_trees].predict(X_test)

            # get effective parameters
            out_iter = track_metrics_through_single_forest(clf, X_train, X_test, y_train, y_test, 0, y_train_resamp, y_train_true)

            next_frame = pd.concat([pd.DataFrame(columns=['iteration', 'seed', 'max_leaf_nodes', 'max_features', 'bootstrap', 'dataset', 'sigma', 'd', 'offset'],
                                                data=[[rep, base_seed, config['max_leaf_nodes'], config['max_features'], config['bootstrap'], marsadd, sigma, d, offset]]),
                                  out_iter], axis=1)

            # write to file
            if save_file:
              for i in range(next_frame.shape[0]):
                next_row = next_frame.iloc[i, :].values
                writer.writerow(next_row)

            # update dataframe
            out_frame = pd.concat([out_frame, next_frame])

        # analyse relative performance of model from each iteration
        for n_trees in range(config['n_estimators']):
          # get predictions of current ensemble size
          preds_n_trees = predictions[:, :(n_trees+1), :].mean(axis=1)

          # get mse for each repetition
          mses = np.mean((preds_n_trees - y_test.reshape(-1, 1))**2, axis=0)

          # avg mse
          avg_mse = np.mean(mses)

          # find best prediction function among n_reps
          mu_star = np.argmin(mses)
          rep_bias = mean_squared_error(preds_n_trees[:, mu_star], y_test)

          mean_excess = np.mean(mses - rep_bias)

          # aggregate to mod_var
          mod_var = np.mean((preds_n_trees - preds_n_trees[:, mu_star].reshape(-1, 1))**2)

          next_summary = pd.DataFrame(columns = HEADER_DECOMP_SUMMARY,
                                      data = [[base_seed, n_trees, config['max_leaf_nodes'], config['max_features'], config['bootstrap'],
                                              marsadd, sigma, d,  offset, avg_mse, rep_bias, mod_var, mean_excess]])
          summary_frame = pd.concat([summary_frame, next_summary])

  if save_file:
    out_file.close()

    # also save summary
    summary_file, writer_summary = create_file_and_writer(file_name, res_dir, HEADER_DECOMP_SUMMARY)

    for i in range(summary_frame.shape[0]):
      next_row = summary_frame.iloc[i, :].values
      writer_summary.writerow(next_row)

    summary_file.close()
  
  return summary_frame


# REAL DATA EXPERIMENTS ------------------
# real data experiments from the appendix
def run_real_data_experiments(file_name, configs, dataset_id, classification: bool = False, decomp_experiment: bool = False, 
                              n_reps=50, n_seeds=10,
                              save_file=True, res_dir='results/', verbose=True, n_train=2000, n_test=2000):
  # create file
  out_file, writer = create_file_and_writer(file_name, res_dir, HEADER_REAL)
  out_frame = pd.DataFrame(columns=HEADER_REAL)

  if decomp_experiment:
    summary_frame = pd.DataFrame(columns=HEADER_DECOMP_REAL_SUMMARY)
  else:
    # no need to fit multiple forests on the same data
    n_reps = 1

  random_states = np.arange(1, n_seeds+1)

  # load data
  X, y = get_openml_data(dataset_id, classification)
  n_total, d = X.shape

  if dataset_id == 45019:
    # bioresponse dataset is smaller
    n_test = 1400 

  # check data_size: reduce size of test set if necessary
  if n_total < (n_test + n_train):
    n_test = n_total - n_train
  
  for base_seed in random_states:
    # split data
    X_train, y_train, X_test, y_test = subsample_and_split_data(X, y, n_train, n_test, base_seed)

    for idx, config in enumerate(configs): # loop through different settings
      if decomp_experiment:
        # store predictions
        predictions = np.zeros(shape=(n_test, config['n_estimators'], n_reps))

      for rep in range(n_reps):
        # fit multiple versions of the same forest on the same data only for decomposition experiment, else just once
        if verbose:
          print("Running experiment with seed {} version {}, bootstrap {}, n_estimators {}, max_features {} and max_leaf_nodes {}.".format(base_seed, rep,
                                                                                                            config['bootstrap'],
                                                                                            config['n_estimators'],
                                                                                              config['max_features'],
                                                                                            config['max_leaf_nodes']))
        
        clf = RandomForestRegressor(n_estimators=config['n_estimators'],
                                  max_leaf_nodes=config['max_leaf_nodes'],
                                  bootstrap=config['bootstrap'],
                                  max_features=config['max_features'],
                                  random_state = base_seed*(rep+1), **RF_PARAMS)

        clf.fit(X_train, y_train)

        if decomp_experiment:
          for n_trees in range(config['n_estimators']):
            # save current predictions
            predictions[:, n_trees, rep] = clf.estimators_[n_trees].predict(X_test)

        # get effective parameters
        out_iter = track_metrics_through_single_forest(clf, X_train, X_test, y_train, y_test, 0, None, None)

        next_frame = pd.concat([pd.DataFrame(columns=['iteration', 'seed', 'max_leaf_nodes', 'max_features', 'bootstrap', 'dataset', 'classification', 'n_train', 'n_test'],
                                            data=[[rep, base_seed, config['max_leaf_nodes'], config['max_features'], config['bootstrap'], dataset_id, classification, n_train, n_test]]),
                              out_iter], axis=1)

        # write to file
        if save_file:
          for i in range(next_frame.shape[0]):
            next_row = next_frame.iloc[i, :].values
            writer.writerow(next_row)

        # update dataframe
        out_frame = pd.concat([out_frame, next_frame])

      if decomp_experiment:
        # analyse relative performance of model from each iteration
        for n_trees in range(config['n_estimators']):
          # get predictions of current ensemble size
          preds_n_trees = predictions[:, :(n_trees+1), :].mean(axis=1)

          # get mse for each repetition
          mses = np.mean((preds_n_trees - y_test.reshape(-1, 1))**2, axis=0)

          # avg mse
          avg_mse = np.mean(mses)

          # find best prediction function among n_reps
          mu_star = np.argmin(mses)
          rep_bias = mean_squared_error(preds_n_trees[:, mu_star], y_test)

          mean_excess = np.mean(mses - rep_bias)

          # aggregate to mod_var
          mod_var = np.mean((preds_n_trees - preds_n_trees[:, mu_star].reshape(-1, 1))**2)

          next_summary = pd.DataFrame(columns = HEADER_DECOMP_REAL_SUMMARY,
                                      data = [[base_seed, n_trees, config['max_leaf_nodes'], config['max_features'], config['bootstrap'],
                                              dataset_id, classification, n_train, n_test, avg_mse, rep_bias, mod_var, mean_excess]])
          summary_frame = pd.concat([summary_frame, next_summary])

  if save_file:
    out_file.close()

    if decomp_experiment:
      # also save summary
      summary_file, writer_summary = create_file_and_writer(file_name, res_dir, HEADER_DECOMP_REAL_SUMMARY)

      for i in range(summary_frame.shape[0]):
        next_row = summary_frame.iloc[i, :].values
        writer_summary.writerow(next_row)

      summary_file.close()
  
  if decomp_experiment:
    return summary_frame
  else:
    return out_frame












