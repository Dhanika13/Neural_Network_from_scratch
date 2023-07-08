import numpy as np 
from neural_network import NeuralNetworksClassifier, TrainTestSplit
import matplotlib.pyplot as plt

# CASE 1: Classification case for spiral data
# -------------------------------------------

# Function to create spiral data
def create_data(n_samples, classes, noise_sig, random_state):
    np.random.seed(random_state)

    X = np.zeros((n_samples*classes, 2))
    y = np.zeros(n_samples*classes, dtype='uint8')

    for class_number in range(classes):
        i = range(n_samples*class_number, n_samples*(class_number+1))
        r = np.linspace(0., 1., n_samples)
        t = np.linspace(class_number*4, (class_number+1)*4, n_samples) \
            + np.random.randn(n_samples)*noise_sig
        X[i] = np.c_[r* np.sin(t*2.5), r*np.cos(t*2.5)]
        y[i] = class_number
    
    return X, y

# Generate input data
X, y = create_data(n_samples=200,
                   classes=3,
                   noise_sig=0.2,
                   random_state=42)

# Split the data into train dataset and test dataset
X_train, y_train, X_test, y_test = TrainTestSplit(shuffle=True).split(X, y)

nn = NeuralNetworksClassifier(learning_rate=1e-1,
                              hidden_layer_sizes=(128,64),
                              max_iter=3001,
                              use_momentum=True,
                              is_decay=True,
                              is_loss_plot=True)

# Fit the data
nn.fit(X_train,y_train)

# Predict on test data
y_pred = nn.predict(X_test)
print(f'accuracy for test data: {np.mean(y_test==y_pred):.3f}')

# Generate collocation points for contour plot
n_x1 = 50 # Number of points for x1
n_x2 = 50 # Number of points for x2

n_contour = n_x1 * n_x2 # Number of collocation points

# Create grid x1 and x2
x1_contour, x2_contour = np.meshgrid(
    np.linspace(-1, 1, n_x1),
    np.linspace(-1, 1, n_x2)
)

# Set X_contour
X_contour = np.concatenate(
    (x1_contour.reshape((n_contour,1)),
    x2_contour.reshape((n_contour,1))),
    axis=1
)

# Predict class for X_contour
y_contour = nn.predict(X_contour)
y_contour = y_contour.reshape((n_x2, n_x1))

# Create plot
train_plot = plt.scatter(X_train[:,0], X_train[:,1], 
                         c=y_train, 
                         cmap='brg', 
                         label='train set')
test_plot = plt.scatter(X_test[:,0], X_test[:,1], 
                        c=y_test, 
                        cmap='brg', 
                        marker='x', 
                        label='test_set')
plt.contourf(x1_contour, x2_contour, y_contour, cmap='brg', alpha=0.2) 
plt.legend(handles=[train_plot, test_plot])
plt.show()