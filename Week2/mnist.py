import tensorflow as tf
import numpy as np

# ------------------------------------------------------------------------------
#                            Load MNIST Data
# ------------------------------------------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')

n_inputs = 28 * 28
n_outputs = 10

# ------------------------------------------------------------------------------
#                          Create Network
# ------------------------------------------------------------------------------
tf.reset_default_graph()


# function to create a network
def dnn(inputs, hidden_layers=[100], name=None, activation=tf.nn.relu,
        initializer=None):

    with tf.variable_scope(name, 'dnn'):
        for i in range(len(hidden_layers)):
            inputs = tf.layers.dense(inputs, hidden_layers[i],
                                     activation=activation,
                                     kernel_initializer=initializer,
                                     name='hidden%d' % (i + 1))
    return inputs


# create the placeholders for minibatches
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

# create the network
he_init = tf.contrib.layers.variance_scaling_initializer()
logits = dnn(X, hidden_layers=[200, 200, 200], initializer=he_init)

# create the loss function
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name='loss')

# set the optimizer (note, fancier than just plain gradient descent!)
optimizer = tf.train.GradientDescentOptimizer(0.01)
training_op = optimizer.minimize(loss)

# create variable to keep track of losses
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# ------------------------------------------------------------------------------
#                            Do Gradient Descent
# ------------------------------------------------------------------------------

tf.set_random_seed(1234)
np.random.seed(1234)

n_epochs = 30
batch_size = 64

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, 'Train Accuracy: ', acc_train, 'Test Accuracy: ',
              acc_test)
