# Neural Network from scratch using Python

This project has an objective to create a simple code of Machine Learning algorithm from scratch in OOP style. I choose Neural Network algorithm for this project.  

## Table of Contents
1. [What is Neural Network?](#what-is-nn)
2. [Components of Learning](#component-of-learning)
3. [Pseudocode](#pseudocode)
4. [Examples](#examples)

## What is Neural Network? <a name="what-is-nn"/>

Neural Network is one of the popular algorithms in Machine Learning which are inspired by organic brains. This model consists of interconnected nodes, known as neurons, organized into layers. The network's structure is typically divided into three main types of layers: input layer, hidden layers, and output layer. Details of this model will be explained later.

---
## Components of Learning <a name="component-of-learning"/>

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
## Pseudocode <a name="pseudocode"/>

### Solve the model parameters (fitting)
- Input
  - Set of training data with input X and output y
  - Network structure : list of hidden_layer_sizes, activation function
  - Optimization : learning_rate $\alpha$, batch_size, max_iter, decay, momentum $\beta$
  - Regularization : lambda $\lambda$
- Output
  - coef_ : the weight $W$
  - intercept_ : the intercept $b$
- Stopping criterion
  - Maximum iteration (epoch)
- Process
> Initialize parameters: random value by Xavier's method.
>
> Initialize learning rate $\alpha = \alpha_0$ and the velocity of coefficients and intercepts as 0.
> 
> **while** (iter < max_iter):
>> Initialize total loss = 0
>>
>> Update learning rate $\alpha := \alpha / (1 + decay \cdot iter)$
>> 
>> **for** batch_slice in batches **do**
>>> **Forward pass**. 
>>>> $a^{[0]} = X$
>>>> 
>>>> **for** k = 1 to last_layer $L$ - 1 **do**
>>>>> Compute activation value $a^{[k]} = \sigma(Z^{[k]}) = \sigma(W^{[k]^T} a^{[k-1]} + b^{[k]})$
>>>>> 
>>>> Compute batch_loss = $\mathcal{L}(a^{[L-1]})$
>>>
>>> **Backward pass**. 
>>>> Compute gradient for last layer $\partial \mathcal{L}/ \partial Z^{[L-1]} = a^{[L-1]} - y \rightarrow \partial \mathcal{L}/ \partial W^{[L-1]}$  and $\partial \mathcal{L}/ \partial b^{[L-1]}$
>>>> 
>>>> **for** i = $L$ - 1 to 1 **do**
>>>>
>>>>> Compute gradient w.r.t. $Z$: $\partial \mathcal{L}/ \partial Z^{[i-1]} = W^{[i]^T} \partial \mathcal{L}/ \partial Z^{[i]}$
>>>>>
>>>>> Compute gradient w.r.t. $W$: $\partial \mathcal{L}/ \partial W^{[i-1]} = \frac{1}{n_{samples}} \left( a^{[i-1]^T} \partial \mathcal{L}/ \partial Z^{[i-1]} + \lambda W^{[i-1]} \right)$
>>>>>
>>>>> Compute gradient w.r.t. $b$: $\partial \mathcal{L}/ \partial b^{[i-1]} = \frac{1}{n_{samples}} \sum \partial \mathcal{L}/ \partial Z^{[i-1]}$
>>>>> 
>>> Compute total loss += batch_loss
>>>
>>> **Update parameters**.
>>>> Change of coefficients: $dW^{[i]} := \beta \cdot v_W^{[i]} - \alpha \cdot \partial \mathcal{L}/ \partial W^{[i]}$
>>>>
>>>> Change of intercepts: $db^{[i]} := \beta \cdot v_b^{[i]} - \alpha \cdot \partial \mathcal{L}/ \partial b^{[i]}$
>>>>
>>>> Save change of parameters as new velocity: $v_W^{[i]} := dW^{[i]}$ and $v_b^{[i]} := db^{[i]}$
>>>>
>>>> Update value of parameters: $W^{[i]} := W^{[i]} + dW^{[i]}$ and $b^{[i]} := b^{[i]} + db^{[i]}$

### Predict

- Input
  - X : The input of test data set
- Output
  - y_pred : The prediction
- Process
> **Forward pass**. 
>> $a^{[0]} = X$
>> 
>>> **for** k = 1 to last_layer $L$ - 1 **do**
>>>> Compute activation value $a^{[k]} = \sigma(Z^{[k]}) = \sigma(W^{[k]^T} a^{[k-1]} + b^{[k]})$
>>>> 
>>> $y_{pred}$ = $a^{[L-1]}$
>>> 

---
## Examples <a name="examples"/>
