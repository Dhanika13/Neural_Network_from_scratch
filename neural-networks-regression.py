import numpy as np 
from neural_network import NeuralNetworksRegressor
import matplotlib.pyplot as plt

n_samples = 500
X = np.linspace(0., 3., n_samples)
y = np.expand_dims(np.sin(1.+X*X) + 0.1*np.random.randn(n_samples), axis=-1)
y_true = np.sin(1.+X*X) 

X = X.reshape(n_samples, 1)

nn = NeuralNetworksRegressor(learning_rate=1e-1,
                             hidden_layer_sizes=(256,256),
                             max_iter=3001,
                             use_momentum=True,
                             is_decay=True,)

nn.fit(X, y)

y_pred = nn.predict(X)

plt.plot(X,y) 
plt.plot(X,y_true) 
plt.plot(X,y_pred) 
plt.show()