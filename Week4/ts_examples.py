import lstm_models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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
n_steps = 10
y_train = (y - y.mean()) / y.std()
lstm = lstm_models.LSTMRegressor(n_steps=n_steps, n_neurons=20, n_layers=1)
lstm.fit(y_train[:n_train], y_train[n_train:],
         n_epochs=1000, eta=0.01, batch_size=256)
y_forc_scaled = lstm.predict(y_train[n_train - n_steps:n_train], n_test)
y_forc = y.std() * y_forc_scaled + y.mean()

# 5 step ahead forecast
k_ahead = 5
np.c_[y_forc[:k_ahead], y[n_train:n_train + k_ahead]]

# yhats
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, lstm.save_path)
    yhat = []
    for t in range(n_steps, n_train):
        y_forc_scaled = lstm.predict(y_train[t - n_steps:t], 1, sess)
        yhat.append(y.std() * y_forc_scaled + y.mean())

plt.plot(np.array(yhat), 'r*')
plt.plot(y[n_steps:n_train], 'b')
plt.savefig('yhat_ts.png', format='png')
plt.show()

# generate a new sequence based on the old on
y_forc_scaled = lstm.predict(y_train[n_train:n_train + n_steps], 500)
y_forc = y.std() * y_forc_scaled + y.mean()
plt.plot(y_forc)
plt.savefig('new_seq_ts.png', format='png')
plt.show()
