# ------------------------------------------------------------------------------
#   Implementation of RNN + Time Series Examples
# ------------------------------------------------------------------------------


import numpy as np
import pdb


class RNN(object):

    def __init__(self, n_hidden=50):
        self.input_size = input_size
        self.n_hidden = n_hidden
        sigma_Wxh = np.sqrt(1. / input_size)
        sigma_Whh = np.sqrt(1. / n_hidden)
        sigma_Why = np.sqrt(1. / n_hidden)

        self.Wxh = sigma_Wxh * np.random.randn(n_hidden, input_size)
        self.Whh = sigma_Whh * np.random.randn(n_hidden, n_hidden)
        self.Why = sigma_Why * np.random.randn(input_size, n_hidden)
        self.bh = np.zeros((n_hidden, 1))
        self.by = np.zeros((input_size, 1))

    def forward_pass(self, x):
        T = len(x)
        h = np.zeros((T + 1, self.n_hidden))
        h[-1] = np.zeros(self.n_hidden)
        yhat = np.zeros((T, self.input_size))
        for t in xrange(T):
            h[t] = np.tanh(self.Wxh.dot(x[t]) +
                           self.Whh.dot(h[t - 1]) + self.bh)
            yhat[t] = self.Why.dot(h[t]) + self.by
        return yhat, h

    def backward_pass(self, x, y):
        pdb.set_trace()
        T = len(x)
        yhat, h = self.forward_pass(x)
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(
            self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(h[0])
        for t in reversed(xrange(T)):
            dy = yhat - y
            dWhy += np.outer(dy, h[t])
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - h[t] * h[t]) * dh
            dbh += dhraw
            dWxh += np.outer(dhraw, x[t])
            dWhh += np.outer(dhraw, h[t - 1])
            dhnext = np.dot(self.Whh.T, dhraw)

        # clip to mitigate exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -2, 2, out=dparam)

        return dWxh, dWhh, dWhy, dbh, dby, h[T - 1]

    def predict(self, x):
        yhat, _ = self.forward_pass(x)
        return yhat

    def train_sgd(self, X, y, eta=0.01, n_epochs=50):
        '''
        @param X an nxp array
        @param y an nxp array
        '''
        n = len(X)
        for epoch in xrange(n_epochs):
            for i in np.random.permutation(n):
                dWxh, dWhh, dWhy, dbh, dby, _ = self.backward_pass(X[i], y[i])
                self.Wxh += -eta * dWxh
                self.Whh += -eta * dWhh
                self.Why += -eta * dWhy
                self.bh += -eta * dbh
                self.by += -eta * dby
            loss = self.training_loss(X, y)
            print 'Epoch: %d Loss: %f' % (epoch, loss)

    def training_loss(self, X, y):
        '''
        Get the MSE
        '''
        n_seqs, T = X.shape
        assert T == self.seq_length

        mse = 0.0
        for i in xrange(n_seqs):
            yhat, _ = self.predict(X[i])
            mse += np.sum((yhat - y[i])**2)
        return mse

    def make_sample(self, x0):
        '''
        '''
        pass


''
# ------------------------------------------------------------------------------
#                         Time Series Experiments
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.api import arma_generate_sample


# create AR(p) series + fit using MLE
n = 500
y_series = arma_generate_sample([1, -0.5, 0, -0.5], [1], n)
arma = ARMA(y_series, (5, 0)).fit()


# plot
plt.plot(range(len(y_series)), y_series)
plt.show()

# fit using RNN

n_steps = 10
n_obs = n - n_steps
X = np.array([y_series[i:(i + n_steps)] for i in range(n_obs)])
y = np.array([y_series[(i + 1):(i + 1 + n_steps)] for i in range(n_obs)])

rnn = RNN(1, n_hidden=7)
rnn.train_sgd(X, y, eta=0.01, n_epochs=50)

# predict using both
