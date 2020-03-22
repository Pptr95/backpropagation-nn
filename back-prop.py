import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io as sio



def plot_decision_boundary(model, X, y):
    plt.figure(figsize=(8,8))
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0], cmap=plt.cm.Spectral)
    plt.show()


def load_flower_dataset(seed):
    np.random.seed(seed)
    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros((m,D))
    Y = np.zeros((m,1), dtype='uint8')
    a = 4 

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2
        r = a*np.sin(4*t) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y



def sigmoid(z):
    return 1/(1+np.exp(-z))


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initialize_weights(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    weights = {"W1": W1,
               "b1": b1,
               "W2": W2,
               "b2": b2}
    
    return weights


def loss_function(A2, Y):
    m = Y.shape[1] 
    cost = (-1/m) * (np.dot(np.log(A2), Y.T) + np.dot(np.log(1 - A2), (1 - Y).T))
    cost = float(np.squeeze(cost))
    return cost




def predict(weights, X):
    A2, prev_activations = forward_propagation(X, weights)
    predictions = np.where(A2 > 0.5, 1, 0)
    
    return predictions


def update_weights(weights, gradients, learning_rate = 0.1):
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
   
    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]
 
    W1 = W1 - (learning_rate * dW1)
    b1 = b1 - (learning_rate * db1)
    W2 = W2 - (learning_rate * dW2)
    b2 = b2 - (learning_rate * db2)
   
    weights = {"W1": W1,
               "b1": b1,
               "W2": W2,
               "b2": b2}
    
    return weights


def forward_propagation(X, weights):
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
     
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    prev_activations = {"Z1": Z1,
                        "A1": A1,
                        "Z2": Z2,
                        "A2": A2}
    
    return A2, prev_activations



'''
Backprop is the algorithm for determining how a single training example would like to change weights and biases.
Not just in terms of whether they should go up and down, but in terms of what relative proportions to those changes
cause the most rapid decrease of the cost loss function.

Specifically it provides iterative rules to compute partial derivatives of the loss function wrt each weight of the network.
'''
def backward_propagation(weights, prev_activations, X, Y):
    m = X.shape[1]
    
    W1 = weights["W1"]
    W2 = weights["W2"]
  
    A1 = prev_activations["A1"]
    A2 = prev_activations["A2"]

    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
  
    
    gradients = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
    
    return gradients



def NeuralNetwork(X, Y, n_h, num_iterations = 10000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    weights = initialize_weights(X.shape[0], n_h, Y.shape[0])
  
    for i in range(0, num_iterations):
        A2, prev_activations = forward_propagation(X, weights)
        cost = loss_function(A2, Y)
        gradients = backward_propagation(weights, prev_activations, X, Y)
        weights = update_weights(weights, gradients)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return weights




# run model

X, Y = load_flower_dataset(1)
n_h = 4
hidden_layer_sizes = [1, 2, 5]
for i, n_h in enumerate(hidden_layer_sizes):
    weights = NeuralNetwork(X, Y, n_h, num_iterations = 5000, print_cost = True)
    plot_decision_boundary(lambda x: predict(weights, x.T), X, Y)
    predictions = predict(weights, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %\n".format(n_h, accuracy))
