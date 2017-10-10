# show example of softmax / multilayer for spiral example

import numpy as np
import os
import pickle
import pdb
import matplotlib.pyplot as plt


class DNN(object):

    def __init__(self, layer_sizes, activation, activation_prime, lamda,
                 he_init=True):
        '''
        @param layer_sizes [n_input hidden_1 ... hidden_k output]
        '''
        self.n_layers = len(layer_sizes)
        self.hidden_layers = layer_sizes[1:-1]
        self.n_inputs = layer_sizes[0]
        self.n_outputs = layer_sizes[-1]
        self.lamda = lamda
        self.activation = activation
        self.activation_prime = activation_prime

        # initialize weights and biases
        self.biases = [np.zeros((x, 1)) for x in layer_sizes[1:]]
        self.weights = []

        for i_in, i_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            if he_init:
                self.weights.append(
                    np.random.randn(i_out, i_in) / np.sqrt(2 * i_in)
                )
            else:
                self.weights.append(0.01 * np.random.randn(i_out, i_in))

    def predict(self, X, return_probs=False):
        assert X.shape[0] == self.n_inputs, 'Check that X has right shape'

        out = X
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            out = self.activation(np.dot(w, out) + b)

        scores = np.dot(self.weights[-1], out) + self.biases[-1]
        probs = self.softmax(scores)

        if return_probs:
            return probs
        else:
            n = X.shape[1]
            class_dense = np.argmax(probs, axis=0)
            class_sparse = np.zeros_like(probs)
            class_sparse[class_dense, range(n)] = 1
            return class_sparse

    def compute_loss_gradient(self, X, y):
        assert X.shape[0] == self.n_inputs, 'Check that X has right shape'
        assert y.shape[0] == self.n_outputs, 'Check that y has right shape'
        n_examples = X.shape[1]

        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        ## forward pass ##

        # hidden layers
        activations = [X]
        zs = [X]
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(self.activation(z))

        # the output layer
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        probs = self.softmax(z)

        ## backward pass (delta = dL / dz) ##

        # start with the last layer
        delta = probs
        delta[np.argmax(y, axis=0), range(n_examples)] -= 1
        delta /= n_examples

        dw[-1] = np.dot(delta, activations[-1].T)
        db[-1] = np.sum(delta, axis=1, keepdims=True)

        for l in xrange(2, self.n_layers):
            z = zs[-l + 1]
            g_prime = self.activation_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * g_prime

            dw[-l] = np.dot(delta, activations[-l].T)
            db[-l] = np.sum(delta, axis=1, keepdims=True)

        return dw, db

    def train(self, X, y, X_test, y_test, learning_rate=0.1, n_epochs=100,
              batch_size=64):
        for epoch in range(n_epochs):
            mini_batches = self.minibatch_indices(X.shape[1], batch_size)
            for batch in mini_batches:
                dw, db = self.compute_loss_gradient(X[:, batch], y[:, batch])
                for i in xrange(len(self.weights)):
                    self.weights[i] += -learning_rate * dw[i]
                    self.biases[i] += -learning_rate * db[i]
            if epoch % 5 == 0:
                train_acc = self.accuracy(self.predict(X), y)
                test_acc = self.accuracy(self.predict(X_test), y_test)
                out_str = 'Epoch: {}, Train Accuracy: {}, Test Accuracy: {}'
                print out_str.format(epoch, train_acc, test_acc)

    def softmax(self, scores):
        return np.exp(scores) / np.sum(np.exp(scores), axis=0, keepdims=True)

    def accuracy(self, y, yhat):
        '''
        '''
        return np.all(yhat == y, axis=0).mean()

    def minibatch_indices(self, n, batch_size):
        '''
        Generate the indices for the mini-batches
        '''
        ix = np.random.permutation(range(n))
        batches = [ix[k:k + batch_size] for k in xrange(0, n, batch_size)]
        return batches

    def save_model(self, model_path):
        os.mkdir(model_path)
        with open(model_path + '/weights.p', 'wb') as f:
            pickle.dump(self.weights, f)
        with open(model_path + '/biases.p', 'wb') as f:
            pickle.dump(self.weights, f)

    def load_model(self, model_path):
        with open(model_path + '/weights.p', 'rb') as f:
            self.weights = pickle.load(f)
            print 'Loaded saved weights'
        with open(model_path + '/biases.p', 'rb') as f:
            self.biases = pickle.dump(f)
            print 'Loaded saved biases'


''

# ------------------------------------------------------------------------------
#                               Todo
# ------------------------------------------------------------------------------

# make sure that when selecting rows in minibatch, don't get a 1d array
# add in regularization somewhere ... plus make sure it is scaled correctly

# ------------------------------------------------------------------------------
#                               Spiral Example
# ------------------------------------------------------------------------------

## some training data ##


def spiral_data(N, K):
    X = np.zeros((2, N * K))
    y = np.zeros((K, N * K), dtype='uint8')
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
        y[j, ix] = 1
    return X, y


N, K = 100, 3
X, y = spiral_data(N, K)
plt.scatter(X[0, :], X[1, :], c=y.argmax(axis=0), s=40, cmap=plt.cm.Spectral)
plt.show()

x1, x2 = np.meshgrid(np.linspace(-1, 1, 80), np.linspace(-1, 1, 80))
X_test = np.c_[x1.reshape(-1), x2.reshape(-1)].T

## softmax classifier ##

softmax = DNN([2, 3], None, None, 0, False)
softmax.train(X, y, X, y)
yhat = softmax.predict(X_test).argmax(axis=0)

plt.contourf(x1, x2, yhat.reshape((80, 80)), cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[0, :], X[1, :], c=y.argmax(axis=0), s=40, cmap=plt.cm.Spectral)
plt.show()

## depth 3 classifier ##


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    z = np.zeros_like(x)
    z[x > 0] = 1
    return z

nn = DNN([2, 10, 10, 10, 3], relu, relu_prime, 0, True)
nn.train(X, y, X, y, n_epochs=1000)
yhat = nn.predict(X_test).argmax(axis=0)

plt.contourf(x1, x2, yhat.reshape((80, 80)), cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[0, :], X[1, :], c=y.argmax(axis=0), s=40, cmap=plt.cm.Spectral)
plt.show()
