# ------------------------------------------------------------------------------
#                      Classification with the Fashion Data
# ------------------------------------------------------------------------------


import sys
sys.path.insert(0, '../Week5/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utility
from sklearn.model_selection import KFold

# ------------------------------------------------------------------------------
#                                 Define the Model
# ------------------------------------------------------------------------------

height = 28
width = 28
channels = 1
n_inputs = height * width
n_outputs = 10

conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = 'SAME'

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = 'SAME'

pool3_fmaps = conv2_fmaps
n_fcl = 53

utility.reset_graph()

with tf.name_scope('inputs'):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    X_reshaped = tf.reshape(X, shape=(-1, height, width, channels))
    y = tf.placeholder(tf.int32, shape=(None), name='y')

conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps,
                         kernel_size=conv1_ksize,
                         strides=conv1_stride,
                         padding=conv1_pad,
                         activation=tf.nn.relu,
                         name='conv1')

conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps,
                         kernel_size=conv2_ksize,
                         strides=conv2_stride,
                         padding=conv2_pad,
                         activation=tf.nn.relu,
                         name='conv2')

with tf.name_scope('pool3'):
    pool3 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
    pool3_flat = tf.reshape(pool3, shape=(-1, pool3_fmaps * 7 * 7))

with tf.name_scope('fcl'):
    fcl = tf.layers.dense(pool3_flat, n_fcl, activation=tf.nn.relu, name='fcl')

with tf.name_scope('output'):
    logits = tf.layers.dense(fcl, n_outputs, name='output')
    y_prob = tf.nn.softmax(logits, name='y_prob')

with tf.name_scope('train'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                              labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# ------------------------------------------------------------------------------
#                             Get Data
# ------------------------------------------------------------------------------


d_path = '/Users/matthewolson/Documents/Data/Fashion/'
X_train, y_train = utility.read_fashion(range(10), 'training', d_path)
X_test, y_test = utility.read_fashion(range(10), 'testing', d_path)

# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot

# ------------------------------------------------------------------------------
#                             Train Model
# ------------------------------------------------------------------------------

n_epochs = 50
batch_size = 8

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        kf = KFold(n_splits=X_train.shape[0] // batch_size, shuffle=True)
        for fold in kf.split(X_train):
            batch_ix = fold[1]
            X_batch, y_batch = X_train[batch_ix], y_train[batch_ix]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
        save_path = saver.save(sess, "./fashion_model")

# ------------------------------------------------------------------------------
#                             Look at Filters, etc.
# ------------------------------------------------------------------------------

ix = 111
img = X_train[ix].reshape((28, 28))
plt.imshow(img, cmap='gray', interpolation='bessel')
plt.savefig('shoe.png', format='png')
plt.show()

with tf.Session() as sess:
    saver.restore(sess, 'fashion_model')
    # get kernel weights
    conv_kernels_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[0]
    conv_kernels = sess.run(conv_kernels_vars)
    # run image through first layer convs
    conv_img = sess.run(conv1, feed_dict={X: img.reshape((1, -1))})

utility.plot_multiple_images(np.squeeze(conv_img).transpose((2, 0, 1)), 4, 8)
plt.tight_layout()
plt.savefig('shoe_features.png', format='png')
plt.show()


# ------------------------------------------------------------------------------
#                             Compare to RF
# ------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=100, verbose=3)
rf.fit(X_train, y_train)
rf.verbose = False
rf.score(X_test, y_test)

ix = np.random.permutation(len(y_test))[0]
print rf.predict(X_test[ix].reshape((1, -1)))
print y_test[ix]
plt.imshow(X_test[ix].reshape(28, 28), cmap='gray', interpolation='bessel')
plt.show()
