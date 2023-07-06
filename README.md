# Neural Network from scratch using Python

This project has an objective to create a simple code of Machine Learning algorithm from scratch in OOP style. I choose Neural Network algorithm for this project.  

## What is Neural Network?

Neural Network is one of the popular algorithms in Machine Learning which are inspired by organic brains. This model consists of interconnected nodes, known as neurons, organized into layers. The network's structure is typically divided into three main types of layers: input layer, hidden layers, and output layer. Details of this model will be explained later.

---
## Components of Learning 

### Hypothesis
Every neuron on a specific layer receives input of linear lines from the previous layer, then transforms it into non-linear function that we will call the activation function. Thus we will have non-linear hypothesis space.

### Parameters
- Weights or coefficients
- Intercepts

### Learning Algorithm
**Objective: minimizing** the loss function based on the case (classification or regression). This objective can be achieved by using gradient descent with additional algorithms to increase its efficiency. 

Hyperparameter for this algorithm:
- Network structure
  - Number of hidden layers
  - Number of neurons in each layer
  - Activation function
- Learning & optimization
  - Learning rate
  - Batch size
  - Epochs
  - Decay
  - Momentum
- Regularization
  - Lambda

### Prediction
**Output:** most probable class for classification and some value for regression.

---
## Pseudocode

### Solve the model parameters (fitting)
- Input
  - X : The input training data set
  - y : The output training data set
  - hidden_layer_sizes
  - activation
  - batch_size
  - max_iter
  - learning_rate
  - decay
  - momentum
  - random_state
- Output
  - coef_ : the weight
  - intercept_ : the intercept
- Stopping criterion
  - Maximum iteration (epoch)
- Process
> Initialize parameters: random value by Xavier's method
>
> **while** (iter < max_iter):
>> **for** batch_slice in batches:
>>> **Forward pass**. Compute activation value every neuron $a^{[k]} = \sigma(z^{[k]}) = \sigma(w^{[k]^T} a^{[k-1]} + b^{[k]})$
>>>
>>> **Backward pass**. Compute 
