import numpy as np

from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.metrics import mean_squared_error


NODE_NOT_USED = -999

# implementing bootstrap subsampling
def get_bootstrap_weights(random_state, n_samples, max_samples):
  # get the number of times each training instance is used by particular bootstrap sample
  # (based on logic in sklean.ensemble._forest._compute_oob_predictions)

  n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, max_samples)
  sample_indices = _generate_sample_indices(random_state, n_samples, n_samples_bootstrap)

  return np.bincount(sample_indices, minlength=n_samples)

def create_S_from_tree(tree, X_train, X_test, bootstrap_weights=None):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # get node labels
    nodes_train = tree.apply(X_train).reshape((n_train, 1))
    nodes_test = tree.apply(X_test).reshape((n_test, 1))

    nodes_train_used = nodes_train.copy()

    # remove any sample *not* used in bootstrap sample
    if bootstrap_weights is not None:
      nodes_train_used[bootstrap_weights == 0] = NODE_NOT_USED

    # expand to vectorize S creation
    nodes_train_exp = np.repeat(nodes_train_used, n_train, axis=1)
    nodes_test_exp = np.repeat(nodes_train_used, n_test, axis=1)

    # create S on training data
    S_train = np.transpose(nodes_train_exp) == nodes_train

    # create S on test data
    S_test = np.transpose(nodes_test_exp) == nodes_test

    # reweight based on bootstrap sample
    if bootstrap_weights is not None:
      S_train = S_train * bootstrap_weights
      S_test = S_test * bootstrap_weights

    # normalize so rows sum up to 1
    S_train = S_train / S_train.sum(axis=1)[:, None]
    S_test = S_test / S_test.sum(axis=1)[:, None]

    return S_train, S_test


def create_S_from_forest(forest, X_train, X_test):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(forest.estimators_)

    # initialise
    S_all_train = np.zeros(shape=(n_train, n_train))
    S_all_test = np.zeros(shape=(n_test, n_train))
    for n in range(n_trees):
        if forest.bootstrap:
          bootstrap_weights = get_bootstrap_weights(forest.estimators_[n].random_state, n_train, forest.max_samples)
        else:
          bootstrap_weights = None
        S_train, S_test = create_S_from_tree(forest.estimators_[n], X_train, X_test, bootstrap_weights=bootstrap_weights)
        S_all_train += S_train
        S_all_test += S_test
    S_all_train = S_all_train / S_all_train.sum(axis=1)[:, None]
    S_all_test = S_all_test / S_all_test.sum(axis=1)[:, None]

    return S_all_train, S_all_test


def create_S_from_single_boosted_tree(tree, S_gb_prev, X_train, X_test):
    S_train, S_test = create_S_from_tree(tree, X_train, X_test)
    if S_gb_prev is None:
        # first tree: just normal tree
        return S_train, S_test

    # reduce contributions where appropriate ----
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # get nodes in tree
    nodes_train = tree.apply(X_train).reshape((n_train, 1))
    nodes_test = tree.apply(X_test).reshape((n_test, 1))

    # get unique nodes
    all_nodes = np.unique(nodes_train)
    n_nodes = len(all_nodes)

    node_corrections = np.zeros(shape=(n_nodes, n_train))
    S_train_correction = np.zeros(shape=(n_train, n_train))
    S_test_correction = np.zeros(shape=(n_test, n_train))

    for i, n in enumerate(all_nodes):
        # create correction matrix
        leaf_id_train = (nodes_train == n).reshape((-1,))
        node_corrections[i, :] = S_gb_prev[leaf_id_train, :].sum(axis=0) / np.sum(leaf_id_train)

        # collect for train examples
        S_train_correction[leaf_id_train, :] = node_corrections[i, :]

        # collect for test examples
        leaf_id_test = (nodes_test == n).reshape((-1,))
        S_test_correction[leaf_id_test, :] = node_corrections[i, :]

    S_train = S_train - S_train_correction
    S_test = S_test - S_test_correction
    return S_train, S_test


def create_S_from_gbtregressor(gbtreg, X_train, X_test, verbose=0):
    # count samples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # count estimators
    n_trees = len(gbtreg.estimators_)
    lr = gbtreg.learning_rate

    S_train_curr = np.zeros(shape=(n_train, n_train))
    S_test_curr = np.zeros(shape=(n_test, n_train))

    for n in range(n_trees):
        if verbose > 0:
            print('Computing S for tree {}.'.format(n))
        S_train, S_test = create_S_from_single_boosted_tree(gbtreg.estimators_[n][0],
                                                            None if n == 0 else S_train_curr,
                                                            X_train, X_test)
        S_train_curr += lr * S_train
        S_test_curr += lr * S_test

    return S_train_curr, S_test_curr