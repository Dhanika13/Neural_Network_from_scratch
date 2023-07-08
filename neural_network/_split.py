import numpy as np 

class TrainTestSplit:
  """ 
  Train-test split

  Split dataset into train and test datset.

  Parameters
  ----------
  test_size : float, default=0.25
    Ratio of test size to original dataset.
  
  shuffle : bool, default=False
    When the value is True, shuffle the indices.

  random_state : int, default=42
      When `shuffle` is True, `random_state` affects the ordering 
      of the indices. Otherwise, this parameter has no effect.
  """
  def __init__(
    self,
    test_size=0.25,
    shuffle=False,
    random_state=42
  ): 
    self.test_size = test_size
    self.shuffle = shuffle
    self.random_state = random_state

  def split(self, X, y):
    """ 
    Parameters
    ----------
    X : {array-like} of shape (n_samples, n_features)
        Input of sample dataset

    y : {array-like} of shape (n_samples, n_outputs)
        Output of sample dataset
    
    Returns
    -------
    X_train : {array-like} of shape (n_train, n_features)
        Input of training dataset

    y_train : {array-like} of shape (n_train, n_outputs)
        Output of training dataset

    X_test : {array-like} of shape (n_test, n_features)
        Input of test dataset

    y_test : {array-like} of shape (n_test, n_outputs)
        Output of test dataset
    """ 
    n_samples = X.shape[0]

    if self.shuffle:
        np.random.seed(self.random_state)
        test_ind = np.random.choice(range(n_samples), int(self.test_size*n_samples))
        test_ind = sorted(test_ind, reverse=False)
    else:
        test_ind = np.arange(n_samples-int(self.test_size*n_samples), n_samples)

    train_ind = list(np.arange(n_samples))
    for i in sorted(test_ind, reverse=True):
        del train_ind[i]
    
    X_train = X[train_ind]
    y_train = y[train_ind]
    X_test = X[test_ind]
    y_test = y[test_ind]

    return X_train, y_train, X_test, y_test