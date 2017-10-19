# ------------------------------------------------------------------------------
#                           RNN in TensorFlow
# ------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.api import arma_generate_sample


# ------------------------------------------------------------------------------
#                             Time Series Examples
# ------------------------------------------------------------------------------

## create a series ##
np.random.seed(123)
n_train = 500
n_forc = 25
y_series = arma_generate_sample([1, -0.5, 0, -0.5], [1], n_train + n_forc)

# something non-stationary
# and more
# fun !!!


## wrap the series into matrices [n x n_steps] ##
n_steps = 10
n_obs = (n_train + n_forc) - n_steps
X_ = np.array([y_series[i:(i + n_steps)] for i in range(n_obs)])
y_ = np.array([y_series[(i + 1):(i + 1 + n_steps)] for i in range(n_obs)])

X_train = X_[:(n_train - n_steps)].reshape(-1, n_steps, 1)
y_train = y_[:(n_train - n_steps)].reshape(-1, n_steps, 1)

X_test = X_[(n_train - n_steps):].reshape(-1, n_steps, 1)
y_test = y_[(n_train - n_steps):].reshape(-1, n_steps, 1)

## fit using RNN ##
n_neurons = 50
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, n_steps, 1])
y = tf.placeholder(tf.float32, [None, n_steps, 1])
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.tanh),
    output_size=1)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(0.001)
training_op = optimizer.minimize(loss)


## train the model ##
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 1000
with tf.Session() as sess:
    init.run()
    for iteration in range(n_epochs):
        sess.run(training_op, feed_dict={X: X_train, y: y_train})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_test, y: y_test})
            print(iteration, "\tMSE:", mse)
    saver.save(sess, "./ts_model")

## retreive the forecasts ##
with tf.Session() as sess:
    saver.restore(sess, "./ts_model")
    yhat_rnn = sess.run(outputs, feed_dict={X: X_train})
    yhat_rnn = np.append(yhat_rnn[:-1, 0, 0], yhat_rnn[-1, :, 0])
    y_forc_rnn = sess.run(outputs, feed_dict={X: X_test})[:, 0, 0]

## make forecasts using an ARMA model ##
arma = ARMA(y_series[:n_train], (5, 0)).fit()
yhat_arma = arma.predict()
y_forc_arma = arma.predict(start=n_train, end=n_train + n_forc - 1)

# make a plot
plt.sublpot(122)
plt.title('ARMA Predictions')
plt.plot(range(n_train + n_forc), y_series, "b-", linewidth=3)
plt.plot(range(n_train), yhat_arma, "r-", linewidth=3)
plt.plot(range(n_train, n_train + n_forc), y_forc_arma, "g-", linewidth=3)

plt.subplot(122)
plt.title('RNN Predictions')
plt.plot(range(n_train + n_forc), y_series, "b-", linewidth=3)
plt.plot(range(n_train), yhat_arma, "r-", linewidth=3)
plt.plot(range(n_train, n_train + n_forc), y_forc_arma, "g-", linewidth=3)


''
# ------------------------------------------------------------------------------
#                            LSTM:
# ------------------------------------------------------------------------------

# use LSTM for ...
