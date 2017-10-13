# ------------------------------------------------------------------------------
#                          DNN in TensorFlow w/ Dropout
# ------------------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
#                             Create a data set
# ------------------------------------------------------------------------------

# mushrooms data
np.random.seed(123)
df = pd.read_csv('mushrooms.csv')
df = pd.get_dummies(df).values
n = df.shape[0]

ix = np.random.permutation(n)
train_ix = ix[:int(0.8 * n)]
test_ix = ix[int(0.8 * n):]

y_train, X_train = df[train_ix, :2], df[train_ix, 2:]
y_test, X_test = df[test_ix, :2], df[test_ix, 2:]


# ------------------------------------------------------------------------------
#                             Some helper functions
# ------------------------------------------------------------------------------


def reset_graph(seed=123):
    '''
    reset tensorflow graph
    '''
    tf.reset_default_graph()
    tf.set_random_seed(seed)


def minibatch_indices(n, batch_size):
    '''
    Generate the indices for the mini-batches
    '''
    ix = np.random.permutation(range(n))
    batches = [ix[k:k + batch_size] for k in xrange(0, n, batch_size)]
    return batches


def accuracy(yhat, y):
    return np.mean(yhat.argmax(1) == y.argmax(1))


# ------------------------------------------------------------------------------
#                    Build one layer network (low level)
# ------------------------------------------------------------------------------

reset_graph()

n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]
hidden = [50]
activation = tf.nn.relu

y = tf.placeholder(tf.int32, shape=(None, n_outputs), name='y')
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')

# first hidden layer
W1 = tf.Variable(initial_value=tf.random_normal((n_inputs, hidden[0])),
                 dtype=tf.float32)
b1 = tf.Variable(initial_value=tf.zeros(hidden[0]), dtype=tf.float32)
z1 = tf.matmul(X, W1) + b1
a1 = activation(z1)

# output layer
W2 = tf.Variable(initial_value=tf.random_normal((hidden[0], n_outputs)),
                 dtype=tf.float32)
b2 = tf.Variable(initial_value=tf.zeros(n_outputs), dtype=tf.float32)
z2 = tf.matmul(a1, W2) + b2


xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=z2)
loss = tf.reduce_mean(xentropy, name='loss')


# training steps
optimizer = tf.train.GradientDescentOptimizer(0.001)
training_op = optimizer.minimize(loss)

# gradient descent, etc.
batch_size = 256
n_epochs = 100

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        batches = minibatch_indices(X_train.shape[0], batch_size)
        for batch in batches:
            sess.run(training_op,
                     feed_dict={X: X_train[batch, :], y: y_train[batch, :]})
        logits = sess.run(z2, feed_dict={X: X_test, y: y_test})
        test_accuracy = accuracy(logits, y_test)
        print 'Epoch: {} Test Accuracy: {}'.format(epoch, test_accuracy)


# ------------------------------------------------------------------------------
#                Build two layer network w/ dropout (higher level)
# ------------------------------------------------------------------------------

reset_graph()

n_inputs = X_train.shape[1]
n_outputs = y_train.shape[1]
hidden = [50, 50]
activation = tf.nn.relu
dropout_rate = 0.5

y = tf.placeholder(tf.int32, shape=(None, n_outputs), name='y')
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')

training = tf.placeholder_with_default(False, shape=(), name='training')
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, hidden[0], activation=tf.nn.relu,
                              name='hidden1')
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, hidden[1], activation=tf.nn.relu,
                              name='hidden2')
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")


with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(0.001)
    training_op = optimizer.minimize(loss)


# gradient descent, etc.
batch_size = 256
n_epochs = 20

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        batches = minibatch_indices(X_train.shape[0], batch_size)
        for batch in batches:
            sess.run(training_op,
                     feed_dict={X: X_train[batch, :], y: y_train[batch, :]})
        logits_ = sess.run(logits, feed_dict={X: X_test, y: y_test})
        test_accuracy = accuracy(logits_, y_test)
        print 'Epoch: {} Test Accuracy: {}'.format(epoch, test_accuracy)
