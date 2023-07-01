import numpy as np 
from neural_network import NeuralNetworksClassifier
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# CASE 1: Classification case for spiral data
# -------------------------------------------

# Generate input data
np.random.seed(42)
X, y = spiral_data(samples=200, classes=3)

nn = NeuralNetworksClassifier(learning_rate=1e-1,
                              hidden_layer_sizes=(128,64),
                              max_iter=3001,
                              use_momentum=True,
                              is_decay=True,
                              alpha=1e-4)

# Fit the data
nn.fit(X,y)

# Generate test data for contour plot
n_x1 = 50 # Number of points for x1
n_x2 = 50 # Number of points for x2

n_test = n_x1 * n_x2 # Number of test data

# Create grid x1 and x2
x1_test, x2_test = np.meshgrid(
    np.linspace(-1, 1, n_x1),
    np.linspace(-1, 1, n_x2)
)

# Set X_test
X_test = np.concatenate(
    (x1_test.reshape((n_test,1)),
    x2_test.reshape((n_test,1))),
    axis=1
)

# Predict class for X_test
y_pred = nn.predict(X_test)

y_pred = y_pred.reshape((n_x2, n_x1))

np.random.seed(123)
X_test2, y_test2 = spiral_data(samples=200, classes=3)

y_pred2 = nn.predict(X_test2)
print(np.mean(y_test2==y_pred2))

# Create plot
# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg') # Plot for input data
plt.scatter(X_test2[:,0], X_test2[:,1], c=y_test2, cmap='brg') # Plot for input data
plt.contourf(x1_test, x2_test, y_pred, cmap='brg', alpha=0.2) # Contour plot for test data
plt.scatter(X_test2[:,0], X_test2[:,1], c=y_pred2, cmap='brg', marker='x') # Plot for test data
plt.show()