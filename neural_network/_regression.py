import numpy as np
from ._base import NeuralNetworks

class NeuralNetworksRegressor(NeuralNetworks):
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
            alpha=alpha,
            random_state=random_state,
        )
        self.is_classifier = is_classifier
    
    def predict(self, X):
        y_pred = self._forward_fast(X)

        return y_pred