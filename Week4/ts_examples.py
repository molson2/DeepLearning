import lstm_models
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
#                        Example 1:
# ------------------------------------------------------------------------------


def ts_1(n, y_minus_1=1, y_minus_2=1):
    '''
    y_k = y_{k-1} + cos(3*y_{k-2})
    '''
    y = np.zeros(n)
    y[0], y[1] = y_minus_2, y_minus_1
    for k in xrange(2, n):
        y[k] = y[k - 1] + np.cos(3 * y[k - 2])
    return y


def ts_2(n):
    '''
    '''
    scale = 0.1
    t = np.linspace(0, scale * n, n)
    return np.cos(np.sqrt(t + 10) * 2 * np.pi)

y = ts_2(3500)

plt.plot(y)
plt.plot(range(3000, 3500), y[3000:], 'r*')
plt.show()

np.random.seed(123)
n_steps = 20
y_train = (y - y.mean()) / y.std()
lstm = lstm_models.LSTMRegressor(n_steps=n_steps, n_neurons=50, n_layers=1)
lstm.fit(y_train[:3000], y_train[3000:],
         n_epochs=100, eta=0.001, batch_size=256)
y_forc_scaled = lstm.predict(y_train[3000 - n_steps:3000], 500)
y_forc = y.std() * y_forc_scaled + y.mean()


k_ahead = 100
plt.plot(y_forc[:k_ahead], 'r-')
plt.plot(y[3000:3000 + k_ahead], 'b*')
plt.show()
