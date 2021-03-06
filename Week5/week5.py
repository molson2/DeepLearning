import tensorflow as tf
import numpy as np
import utility
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
# ------------------------------------------------------------------------------
#                           PCA Example
# ------------------------------------------------------------------------------


def random_rotation(d):
    Q, R = np.linalg.qr(np.random.randn(d, d))
    return Q


def pca(X_data):
    sc = StandardScaler()
    sc.fit(X_data)
    pca = PCA()
    pca.fit(sc.transform(X_data))
    U = pca.fit_transform(sc.transform(X_data))
    return U


def autoencoder2d(X_data, eta=0.01, n_epochs=1000, activation=None):
    n_inputs = X_data.shape[1]
    n_hidden = 100
    n_dim = 2
    sc = StandardScaler()
    sc.fit(X_data)

    if activation is None:
        activation = lambda x: x

    he_init = tf.contrib.layers.variance_scaling_initializer()

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden1 = tf.layers.dense(X, n_hidden, activation=activation,
                              kernel_initializer=he_init)
    hidden2 = tf.layers.dense(hidden1, n_dim, activation=activation,
                              kernel_initializer=he_init)
    hidden3 = tf.layers.dense(hidden2, n_hidden, activation=activation,
                              kernel_initializer=he_init)
    outputs = tf.layers.dense(hidden3, n_inputs)

    loss = tf.reduce_mean(tf.square(outputs - X))
    optimizer = tf.train.AdamOptimizer(eta)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    X_trans = sc.transform(X_data)
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            training_op.run(feed_dict={X: X_trans})
            if epoch % 100 == 0:
                print sess.run(loss, feed_dict={X: X_trans})
        comps = sess.run(hidden2, feed_dict={X: X_trans})
    return comps

np.random.seed(123)
n = 1000
d = 3
X = np.c_[np.random.random((n, 2)), np.zeros((n, d - 2))]
Q = random_rotation(d)
X_data = X.dot(Q.T)

fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter3D(X_data[:, 0], X_data[:, 1], X_data[:, 2])
plt.show()

# do PCA #

U = pca(X_data)
plt.plot(U[:, 0], U[:, 1], 'bo')
plt.show()

# do 'autoencoder' #
utility.reset_graph(123)
U = autoencoder2d(X_data, n_epochs=100, activation=None)
plt.plot(U[:, 0], U[:, 1], 'bo')
plt.show()
