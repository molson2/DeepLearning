import tensorflow as tf
import numpy as np
import utility
import mpl_toolkits.mplot3d.axes3d as p3

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys
# ------------------------------------------------------------------------------
#                           PCA 2d: Example 1
# ------------------------------------------------------------------------------

np.random.seed(123)


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


n = 500
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
utility.reset_graph()
U = autoencoder2d(X_data, n_epochs=100)
plt.plot(U[:, 0], U[:, 1], 'bo')
plt.show()

# ------------------------------------------------------------------------------
#                           PCA 2d: Example 2
# ------------------------------------------------------------------------------

from sklearn.datasets import make_swiss_roll

n = 1000
X_data, t = make_swiss_roll(n)

ix_red = t > 8
ix_blue = t < 8
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.scatter3D(X_data[ix_red, 0], X_data[ix_red, 1], X_data[ix_red, 2], 'ro')
ax.scatter3D(X_data[ix_blue, 0], X_data[ix_blue, 1], X_data[ix_blue, 2], 'bo')
plt.show()

# do pca #
U = pca(X_data)
plt.plot(U[ix_red, 0], U[ix_red, 1], 'ro')
plt.plot(U[ix_blue, 0], U[ix_blue, 1], 'bo')
plt.show()

# do autoencoder #
utility.reset_graph()
U = autoencoder2d(X_data, eta=0.001, n_epochs=5000, activation=None)
plt.plot(U[ix_red, 0], U[ix_red, 1], 'ro')
plt.plot(U[ix_blue, 0], U[ix_blue, 1], 'bo')
plt.show()

# ------------------------------------------------------------------------------
#                           Sparse Autoeoncoder
# ------------------------------------------------------------------------------

utility.reset_graph()
n_inputs = 28 * 28
n_hidden1 = 1000
n_outputs = n_inputs


def kl_divergence(p, q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))

learning_rate = 0.01
sparsity_target = 0.1
sparsity_weight = 0.2

X = tf.placeholder(tf.float32, shape=(None, n_inputs))
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid)
outputs = tf.layers.dense(hidden1, n_outputs)

hidden1_mean = tf.reduce_mean(hidden1, axis=0)
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
loss = reconstruction_loss + sparsity_weight * sparsity_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for epoch in xrange(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
            reconstruction_loss_val, sparsity_loss_val, loss_val = sess.run(
                [reconstruction_loss, sparsity_loss, loss], feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", reconstruction_loss_val,
                  "\tSparsity loss:", sparsity_loss_val, "\tTotal loss:", loss_val)
