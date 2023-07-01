import numpy as np
from ._base import NeuralNetworks

class NeuralNetworksClassifier(NeuralNetworks):
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
        alpha=1e-4,
        random_state=42,
        is_classifier=True,
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
            alpha=alpha,
            random_state=random_state,
        )
        self.is_classifier = is_classifier
    
    def _validate_input(self, X, y):
        n_samples = X.shape[0]

        if len(y.shape) == 1:
            n_classes = np.unique(y)

            y_new = np.zeros((y.shape[0],n_classes.shape[0]))

            for i in n_classes:
                y_new[y==i,i] = 1 
        else:
            y_new = y
            
        return y_new
    
    def _check_accuracy(self, y_true, y_prob):
        class_y_prob = np.argmax(y_prob, axis=1)
        class_y_true = np.argmax(y_true, axis=1)
        accuracy = np.mean(class_y_prob==class_y_true)

        return accuracy
    
    def predict(self, X):
        y_prob = self._forward_fast(X)

        y_pred = np.argmax(y_prob, axis=1)

        return y_pred