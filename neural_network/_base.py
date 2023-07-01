import numpy as np

def relu_activation(Z):
    """ 
    Rectified linear unit function: A(Z) = max(0, Z)
    
    Parameters
    ----------
    Z : array-like of shape (n_samples, n_features)
        The input data.
    """ 
    return np.maximum(Z,0)

def logistic_activation(Z):
    """ 
    Logistic sigmoid function: A(Z) = 1 / (1 + exp(-Z))
    
    Parameters
    ----------
    Z : array-like of shape (n_samples, n_features)
        The input data.
    """ 
    return 1./(1. + np.exp(-Z))

def identity_activation(Z):
    """ 
    Identity function: A(Z) = Z

    Parameters
    ----------
    Z : array-like of shape (n_samples, n_features)
        The input data.
    """ 
    return Z

ACTIVATIONS = {
    "relu": relu_activation,
    "logistic": logistic_activation,
    "identity": identity_activation,
}

def relu_derivative(Z, deltas):
    """ 
    Derivative of relu function: A'(Z) = 1 for Z > 0 and
    A'(Z) = 0 for Z <= 0

    Parameters:
    -----------
    Z : array-like of shape (n_samples, n_features)
        The input data.
    
    deltas : array-like of shape (n_samples, n_features)
        The derivatives of activation function dA/dZ.
    """ 
    grad_Z = np.zeros(Z.shape)
    grad_Z[Z > 0] = 1

    deltas *= grad_Z

def logistic_derivative(Z, deltas):
    """ 
    Derivative of logistic sigmoid function: A'(Z) = Z * (1 - Z)

    Parameters:
    -----------
    Z : array-like of shape (n_samples, n_features)
        The input data.
    
    deltas : array-like of shape (n_samples, n_features)
        The derivatives of activation function dA/dZ.
    """ 
    deltas *= Z*(1-Z)

def identity_derivative(Z, deltas):
    """ 
    Derivative of identity function: A'(Z) = 1

    Parameters:
    -----------
    Z : array-like of shape (n_samples, n_features)
        The input data.
    
    deltas : array-like of shape (n_samples, n_features)
        The derivatives of activation function dA/dZ.
    """ 
    deltas = deltas

DERIVATIVES = {
    "relu": relu_derivative,
    "logistic": logistic_derivative,
    "identity": identity_derivative,
}

def log_loss(y_true, y_prob):
    """ 
    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        Ground truth (correct) labels.
    
    y_prob : array-like of shape (n_samples, n_classes)
        Predicted probalities.

    Returns
    -------
    loss : float
        Value of loss from learning process.
    """ 
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)

    loss = - (y_true * np.log(y_prob)).sum() \
           - ((1 - y_true) * np.log(1 - y_prob)).sum()
    loss /= y_prob.shape[0]

    return loss

def squared_loss(y_true, y_pred):
    """ 
    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        Ground truth (correct) labels.
    
    y_pred : array-like of shape (n_samples, n_output)
        Predicted values.

    Returns
    -------
    loss : float
        Value of loss from learning process.
    """ 
    return 0.5*((y_true -  y_pred)**2).mean()

LOSS_FUNCTIONS = {
    "log_loss": log_loss,
    "squared_error": squared_loss,
}

class NeuralNetworks:
    """
    Neural Networks base class

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
        reach this number iterations.
    
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

    random_state : int, default=42 
        Determines random number generation for parameters initialization.
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
        alpha=1e-3,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.is_decay = is_decay
        self.decay = decay
        self.use_momentum = use_momentum
        self.momentum = momentum
        self.alpha = alpha
        self.random_state = random_state
    
    def _init_coef(self, n_in, n_out):
        """ 
        Initialization of coefficient using method by Xavier Glorot
        ref: https://towardsdatascience.com/xavier-glorot-initialization-in-neural-networks-math-proof-4682bf5c6ec3

        Parameters
        ----------
        n_in : int
            Number of units from i-th layer 
        
        n_out : int
            Number of units from (i+1)-th layer
        
        Returns
        -------
        coef_init : array-like of (n_in, n_out)
            Initial coefficient from i-th layer to (i+1)-th layer
        
        intercept_init : array-like of (n_out,)
            Initial intercept from i-th layer to (i+1)-th layer
        """ 
        # Set random state
        np.random.seed(self.random_state)

        # Bound for uniform distribution
        bound = np.sqrt(6./(n_in + n_out))

        # Initialize coefficient and intercept using uniform distribution
        coef_init = np.random.uniform(
            -bound, bound, (n_in, n_out)
        )
        intercept_init = np.random.uniform(-bound, bound, n_out)

        return coef_init, intercept_init

    def _initalize_parameters(self, layer_units):
        """ 
        Initialize all parameters (coefficients and intercepts) of Nerual Networks
        and set as attributes.

        Parameters
        ----------
        layer_units: array-like of (n_layers,)
            The i-th element represents the number of units in the i-th layer, 
            first layer represents the number of input features and last layer
            represents the number of output features.
        """ 
        # Compute the number of layers
        self.n_layers_ = len(layer_units)

        # Initialize coefficient and intercept layers
        self.coefs_ = []
        self.intercepts_ = []

        for i in range(self.n_layers_ - 1):
            coef_init, intercept_init = self._init_coef(
                layer_units[i],  layer_units[i+1]
            )
            self.coefs_.append(coef_init)
            self.intercepts_.append(intercept_init)
    
    def _generate_batches(self, n_samples, batch_size):
        """ 
        Generate slices containing minibatch elements from input data.
        Last slice may contain less than 'batch_size' elements.

        Parameters
        ----------
        n_samples : int
            Number of samples from input data
        
        batch_size : int
            Size of minibatch.
        
        Yields
        ------
        slice of 'batch_size' elements
        """ 
        start = 0
        for i in range(n_samples // batch_size):
            end = start + batch_size
            if end > n_samples:
                continue
            yield slice(start, end)
            start = end 
        if start < n_samples:
            yield slice(start, n_samples)
    
    def _forward_prop(self, activations):
        """ 
        Forward propagation to compute values of output from every units in 
        neural networks.

        For every layer, perform linear transformation Z = X * W + b and use
        Z as input for activation function A(Z).

        Parameters
        ----------
        activations : list, length = n_layers 
            List where contains output of activation functions from every layer.
        """ 
        # Activation function for hidden layers
        activation_func = ACTIVATIONS[self.activation]

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            # Linear transformation
            activations[i+1] = np.dot(activations[i], self.coefs_[i]) 
            activations[i+1] += self.intercepts_[i]

            # Activation function
            if (i+1) != (self.n_layers_ - 1):
                activations[i+1] = activation_func(activations[i+1])
        
        # Output activation function for last layer
        output_activation_func = ACTIVATIONS[self.out_activation_]
        activations[i+1] = output_activation_func(activations[i+1])
    
    def _forward_fast(self, X):
        """ 
        Forward propagation to predict the output without record activation 
        from every layer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        y_pred : array of shape (n_samples, n_outputs)
            The prediction from input data.
        """ 
        # Initialize first layer
        activation = X

        # Activation function for hidden layers
        activation_func = ACTIVATIONS[self.activation]

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            # Linear transformation
            activation = np.dot(activation, self.coefs_[i]) 
            activation += self.intercepts_[i]

            # Activation function
            if (i+1) != (self.n_layers_ - 1):
                activation = activation_func(activation)
        
        # Output activation function for last layer
        output_activation_func = ACTIVATIONS[self.out_activation_]
        y_pred = output_activation_func(activation)

        return y_pred

    def _compute_grad(self, layer, n_samples, activations, deltas, coef_grads, intercept_grads):
        """ 
        Compute the gradient of loss with respect to coefficients and intercepts 
        for one specified layer.

        dL/dW = (1/m) * A * dL/dZ
        dL/db = (1/m) * dL/dZ

        Parameters
        ----------
        layer : int
            Number of current layer to compute its gradient.
        
        n_samples : int
            Number of samples.
        
        activations : list, length = n_layers 
            List where contains output of activation functions from every layer.
        
        deltas : list, length = n_layers - 1
            List where contains dL/dZ from every layer, except 1st layer.
        
        coef_grads = list, length = n_layers - 1
            List where contains dL/dW from every layer, except 1st layer.
        
        intercept_grads = list, length = n_layers - 1
            List where contains dL/db from every layer, except 1st layer.
        """ 
        coef_grads[layer] = np.dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= n_samples

        intercept_grads[layer] = np.mean(deltas[layer], 0)
    
    def _back_prop(self, X, y, activations, deltas, coef_grads, intercept_grads):
        """ 
        Compute loss function and its derivatives with respect to each parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array of shape (n_samples,)
            The target values
        
        activations : list, length = n_layers 
            List where contains output of activation functions from every layer.
        
        deltas : list, length = n_layers - 1
            List where contains dL/dZ from every layer, except 1st layer.
        
        coef_grads = list, length = n_layers - 1
            List where contains dL/dW from every layer, except 1st layer.
        
        intercept_grads = list, length = n_layers - 1
            List where contains dL/db from every layer, except 1st layer.

        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """ 
        n_samples = X.shape[0]

        # Forward propagation
        self._forward_prop(activations)
        
        # Compute loss function
        loss = LOSS_FUNCTIONS[self.loss_func_](y, activations[-1])

        # Add L2 regularization term to  loss
        regularization_loss = 0
        for s in self.coefs_:
            s = s.ravel()
            regularization_loss += np.dot(s, s)
        loss += (0.5 * self.alpha) * regularization_loss / n_samples

        # Start backward propagation from last layer
        last = self.n_layers_ - 2
        
        # Compute dL/dZ for last layer
        deltas[last] = activations[-1] - y

        # Compute dL/dW and dL/db for last layer
        self._compute_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        # Iteration over the hidden layers 
        derivative_func = DERIVATIVES[self.activation]
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = np.dot(deltas[i], self.coefs_[i].T)
            derivative_func(activations[i], deltas[i-1])

            self._compute_grad(
                i-1, n_samples, activations, deltas, coef_grads, intercept_grads
            )
        
        return loss, coef_grads, intercept_grads
    
    def _update_learning_rate(self, iteration):
        """ 
        Update learning rate if using decay method.
        
        Parameters
        ----------
        iteration: int
            Number of current iteration.
        """ 
        if self.is_decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * iteration))

    def _update_param(self, coef_grads, intercept_grads):
        """ 
        Update parameters (coefficients and intercepts) from its gradients.

        Parameters
        ----------
        coef_grads = list, length = n_layers - 1
            List where contains dL/dW from every layer, except 1st layer.
        
        intercept_grads = list, length = n_layers - 1
            List where contains dL/db from every layer, except 1st layer.
        """ 
        update_coefs = [
            self.momentum * velocity - self.current_learning_rate * grad
            for velocity, grad in zip(self.velocity_coefs_, coef_grads)
        ]
        self.velocity_coefs_ = update_coefs
        
        update_intercepts = [
            self.momentum * velocity - self.current_learning_rate * grad
            for velocity, grad in zip(self.velocity_intercepts_, intercept_grads)
        ]
        self.velocity_intercepts_ = update_intercepts

        for coef, update_coef in zip(self.coefs_, update_coefs):
            coef += update_coef 
        
        for intercept, update_intercept in zip(self.intercepts_, update_intercepts):
            intercept += update_intercept

    def fit(self, X, y):
        """
        Fit the neural network method from the training dataset

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data

        y : {array-like} of shape (n_samples, n_outputs)
            Target values 
        """
        # Check classifier or regression
        if self.is_classifier:
            self.out_activation_ = "logistic"
            self.loss_func_ = "log_loss"
            y = self._validate_input(X, y)
        else:
            self.out_activation_ = "identity"
            self.loss_func_ = "squared_error"

        X = np.array(X).copy()
        y = np.array(y).copy()

        n_samples, n_features = X.shape
        n_outputs = y.shape[1]

        # Create list of layer units
        hidden_layer_sizes = list(self.hidden_layer_sizes)

        layer_units = [n_features] + hidden_layer_sizes + [n_outputs]

        # Initialize coefficients and intercepts
        self._initalize_parameters(layer_units)
        
        # Initialize activations : list for A(Z)
        activations = [X] + [None] * (len(layer_units) - 1)

        # Initialize deltas : list for dL/dZ
        deltas = [None] * (len(layer_units) - 1)
        
        # Initialize coef_grads : list for dL/dW
        coef_grads = [
            np.zeros((n_in_, n_out_)) 
            for n_in_, n_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        # Initialize intercept_grads : list for dL/db
        intercept_grads = [
            np.zeros(n_out_) for n_out_ in layer_units[1:]
        ]
        
        # Set batch size
        if self.batch_size == 'auto':
            batch_size = min(200, n_samples)
        else:
            batch_size = self.batch_size
        
        # Initialize list for velocity  
        self.velocity_coefs_ = [np.zeros_like(coefs) for coefs in self.coefs_]
        self.velocity_intercepts_ = [np.zeros_like(intercepts) for intercepts in self.intercepts_]

        # Set momentum parameter
        if self.use_momentum is False:
            self.momentum = 0.

        # Initialize learning rate 
        self.current_learning_rate = self.learning_rate

        # Initialize list for loss and accuracy
        self.loss_ = []
        self.accuracy_ = []

        # Iteration
        for it in range(self.max_iter):
            # Initialize total loss and total accuracy
            total_loss = 0.0
            total_accuracy = 0.0

            # Update learning rate
            self._update_learning_rate(it)

            # Iterate over minibatch
            for batch_slice in self._generate_batches(n_samples, batch_size):
                X_batch = X[batch_slice]
                y_batch = y[batch_slice]
            
                activations[0] = X_batch
                # Back propagation
                batch_loss, coef_grads, intercept_grads = self._back_prop(
                    X_batch,
                    y_batch,
                    activations,
                    deltas,
                    coef_grads,
                    intercept_grads
                )
                total_loss += batch_loss * (batch_slice.stop - batch_slice.start)

                # Compute total accuracy for classification case
                if self.is_classifier:
                    batch_accuracy = self._check_accuracy(y_batch, activations[-1])
                    total_accuracy += batch_accuracy * (batch_slice.stop - batch_slice.start)

                # Update coefficients and intercepts
                self._update_param(coef_grads, intercept_grads)

            self.loss_.append(total_loss / n_samples)
            self.accuracy_.append(total_accuracy / n_samples)

            # Print the progress
            if (it%100) == 0:
                if self.is_classifier:
                    print(f'epoch: {it}, ' +
                        f'loss: {self.loss_[it]:.3f}, ' +
                        f'acc: {self.accuracy_[it]:.3f}')
                else:
                    print(f'epoch: {it}, ' +
                        f'loss: {self.loss_[it]:.3f}, ')