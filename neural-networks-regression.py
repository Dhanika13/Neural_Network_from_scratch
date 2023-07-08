import numpy as np 
from neural_network import NeuralNetworksRegressor,TrainTestSplit
import matplotlib.pyplot as plt

# CASE 1: Regression case for sine data
# -------------------------------------

# Createe sample data set 
n_samples = 1000

np.random.seed(42)

X = np.linspace(0., 10., n_samples).reshape(n_samples, 1)
y = np.sin(2.*X) + 0.2*np.random.randn(n_samples).reshape(n_samples, 1)
y_true = np.sin(2.*X) 

# Split the data into train dataset and test dataset
X_train, y_train, X_test, y_test = TrainTestSplit(shuffle=True).split(X, y)

nn = NeuralNetworksRegressor(learning_rate=1e-2,
                             hidden_layer_sizes=(128,64),
                             max_iter=6001,
                             use_momentum=True,
                             is_decay=True,
                             is_loss_plot=True)

# Fit the data
nn.fit(X_train, y_train)

# Predict on test data
y_pred = nn.predict(X_test)
print(f'loss for test data: {0.5*((y_test-y_pred)**2).mean():.3f}')

# Predict on every X for plot the prediction function
y_plot = nn.predict(X)

# Plot the solution
plt.scatter(X, y, alpha=0.2, c='b', label='dataset')
plt.plot(X, y_true, 'k-', label='true function') 
plt.plot(X, y_plot, 'r-', label='predicted function') 
plt.legend()
plt.show()