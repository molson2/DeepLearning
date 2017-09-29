import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
#                             Getting Started
# ------------------------------------------------------------------------------

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = y * x**2 + y + 2

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print f.eval()

## OR ##
sess = tf.InteractiveSession()
init.run()
print f.eval()
sess.close()
##

tf.reset_default_graph()

x_data = np.random.randn(10, 3)
x = tf.placeholder(dtype=tf.float32, shape=(None, 3))
beta = tf.constant(np.array([1, 2, 3]), dtype=tf.float32, shape=(3, 1))
y = tf.matmul(x, beta)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    y_even = sess.run(y, feed_dict={x: x_data[::2, :]})
    y_all = sess.run(y, feed_dict={x: x_data})


# ------------------------------------------------------------------------------
#                        Regression 1 (Normal Equations)
# ------------------------------------------------------------------------------

np.random.seed(123)
n = 100
p = 2

beta = np.random.random((p + 1, 1))
X_data = np.c_[np.ones(n), np.random.randn(n, p)]
y_data = X_data.dot(beta) + 0.2 * np.random.randn(n, 1)
X = tf.constant(X_data, dtype=tf.float32, name='X')
y = tf.constant(y_data, dtype=tf.float32, name='y')

beta_hat = tf.matmul(tf.matmul(tf.matrix_inverse(
    tf.matmul(tf.transpose(X), X)), tf.transpose(X)), y)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print beta_hat.eval()


# ------------------------------------------------------------------------------
#                 Computing Derviatives via Autodiff + Optimization
# ------------------------------------------------------------------------------

tf.reset_default_graph()

# consider f(x) = 0.2*(x-1)^2 - x*sin(x-5)
t = np.linspace(-10, 10, 100)
plt.plot(t, 0.2 * (t - 1)**2 - t * np.sin(t - 5))
plt.show()


def f(x):
    return 0.2 * (x - 1)**2 - x * tf.sin(x - 5)

# using tensorflow to compute numerical derivative at x = 0.3
EPS = 1e-5
x = tf.Variable(0, dtype=tf.float32, name='x')
grad = tf.gradients(f(x), x)[0]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    sess.run(tf.assign(x, 0.3))
    tf_grad = grad.eval()
    richardson_grad = sess.run((f(0.3 + EPS) - f(0.3 - EPS)) / (2 * EPS))


# use tensorflow to minimize function
n_iter = 50
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
training_op = optimizer.minimize(f(x))


with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        sess.run(training_op)
        f_val = sess.run(f(x))
        print 'Iter: {} Fval: {}'.format(i, f_val)
    x_star = x.eval()


# ------------------------------------------------------------------------------
#                        Regression 2 (Mini-Batch Gradient Descent)
# ------------------------------------------------------------------------------

tf.reset_default_graph()

# the data
np.random.seed(123)
n = 100
p = 2

beta = np.random.random((p + 1, 1))
X_data = np.c_[np.ones(n), np.random.randn(n, p)]
y_data = X_data.dot(beta) + 0.2 * np.random.randn(n, 1)

# placeholders and variables
X = tf.placeholder(dtype=tf.float32, shape=(None, p + 1))
y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
beta_hat = tf.Variable(tf.random_uniform([p + 1, 1], -1, 1))

# optimization operations
loss = tf.reduce_mean(tf.square(y - tf.matmul(X, beta_hat)))
optimizer = tf.train.GradientDescentOptimizer(0.01)
training_op = optimizer.minimize(loss)

# train with mini-batch gradient descent
n_epochs = 50
batch_size = 10

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        batches = np.arange(0, n).reshape(-1, n // batch_size)
        for batch in batches:
            sess.run(training_op, feed_dict={
                     X: X_data[batch, :], y: y_data[batch]})
        full_loss = sess.run(
            loss, feed_dict={X: X_data[batch, :], y: y_data[batch]})
        print 'Iter: {} Loss {}'.format(epoch, full_loss)
    beta_out = beta_hat.eval()


# ------------------------------------------------------------------------------
#                                 Sparse PCA
# ------------------------------------------------------------------------------


from skimage import io
img = io.imread('mit.jpg', as_grey=True)

# original image
plt.imshow(img, cmap='gray')
plt.show()

# PCA
a, b, c = np.linalg.svd(img)

# sparse PCA
n, m = img.shape
n_comps = 10
gamma = 20

tf.reset_default_graph()
tf.set_random_seed(1234)

A = tf.Variable(tf.random_uniform(
    [n, n_comps], -1, 1), dtype=tf.float32, name='A')
B = tf.Variable(tf.random_uniform(
    [n_comps, m], -1, 1), dtype=tf.float32, name='B')

loss = tf.reduce_sum(tf.square(img - tf.matmul(A, B)))
penalty = gamma * tf.reduce_sum(tf.abs(A)) + gamma * tf.reduce_sum(tf.abs(B))

optimizer = tf.train.GradientDescentOptimizer(0.001)
training_op = optimizer.minimize(loss + penalty)

# train with mini-batch gradient descent
n_epochs = 500

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op)
        print loss.eval()
    A_hat = A.eval()
    B_hat = B.eval()

plt.imshow(np.matmul(A_hat, B_hat), cmap='gray')
plt.show()
