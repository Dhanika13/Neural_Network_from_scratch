import numpy as np
from ._base import NeuralNetworks

class NeuralNetworksRegressor(NeuralNetworks):
    """
    Class for regression case of Neural Network

    Parameters
    ----------
    hidden_layer_sizes : array-like of (n_layer - 2,), default=(100,)
        The i-th element represents the number of units in the i-th hidden layer.
    
    activation : {'relu', 'logistic', 'identity'}, default='relu'
        Activation function for the hidden layer.

        - 'relu' : rectified linear unit function
          returns f(x) = max(0, x)
        
        - 'logistic' : logistic sigmoid function
          returns f(x) = 1 / (1 + exp(-x))
        
        - 'identity' : identity function
          returns f(x) = x
    
    batch_size : int, default='auto'
        Size of minibatches. 
        For 'auto', batch_size=min(200, n_samples)

    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence or
        reach this number iterations. In this case, this is max epoch.
    
    learning_rate : float, default=1e-3
        Value of learning rate to update the parameters.
    
    is_decay : bool, default=False
        When the value is True, decrease value learning rate for every iteration.
    
    decay : float, default=1e-3
        Value for decay parameter. New learning rate follows these rule:
        current_learning_rate = learning_rate / (1 + decay * iteration)

    use_momentum : bool, default=False
        When the value is True, use momentum approach to update the parameters.
    
    momentum : float, default=0.9
        Momentum parameter for update the parameters. Should be between 0 and 1.

    lambda_r : float, default=1e-4
        Regularization parameter.

    random_state : int, default=42 
        Determines random number generation for parameters initialization.

    is_loss_plot : bool, default=False
        When the value is True, plot the loss vs epochs.
        
    is_classifier : bool, default=False
        Output layer using identity activation function.
    """ 
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation='relu',
        batch_size='auto',
        max_iter=200,
        learning_rate=1e-3,
        is_decay=False,
        decay=1e-3,
        use_momentum=False,
        momentum=0.9,
        lambda_r=1e-4,
        random_state=42,
        is_loss_plot=False,
        is_classifier=False,
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            batch_size=batch_size,
            max_iter=max_iter,
            learning_rate=learning_rate,
            is_decay=is_decay,
            decay=decay,
            use_momentum=use_momentum,
            momentum=momentum,
            lambda_r=lambda_r,
            random_state=random_state,
            is_loss_plot=is_loss_plot,
        )
        self.is_classifier = is_classifier
    
    def predict(self, X):
        """ 
        Predict values from test data set. 

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Test data set

        Returns
        -------
        y_pred : {array-like} of shape (n_samples, n_outputs)
            Prediction values from test data set
        """ 
        y_pred = self._forward_fast(X)

        return y_pred