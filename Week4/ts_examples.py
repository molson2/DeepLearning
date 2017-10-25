import lstm_models
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------------------
#                               Example 1
# ------------------------------------------------------------------------------

def ts(n):
    '''
    '''
    scale = 0.01
    t = np.linspace(0, scale * n, n)
    return t * np.sin(t) / 3 + 2 * np.sin(t * 5)


n_train = 1000
n_test = 100
y = ts(n_train + n_test)

plt.plot(y)
plt.plot(range(n_train, n_train + n_test), y[n_train:], 'r*')
plt.show()

np.random.seed(123)
n_steps = 20
y_train = (y - y.mean()) / y.std()
lstm = lstm_models.LSTMRegressor(n_steps=n_steps, n_neurons=20, n_layers=1)
lstm.fit(y_train[:n_train], y_train[n_train:],
         n_epochs=1000, eta=0.05, batch_size=256)
y_forc_scaled = lstm.predict(y_train[n_train - n_steps:n_train], n_test)
y_forc = y.std() * y_forc_scaled + y.mean()

# 10 step ahead forecast
k_ahead = 10
np.c_[y_forc[:k_ahead], y[n_train:n_train + k_ahead]]


# generate a new sequence based on the old on
y_forc_scaled = lstm.predict(np.random.randn(n_steps), 100)
y_forc = y.std() * y_forc_scaled + y.mean()
plt.plot(y_forc)
plt.show()
