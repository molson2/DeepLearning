import sys
sys.path.insert(0, '../Week5/')
import utility
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from sklearn.model_selection import KFold
inception_path = 'inception_v3.ckpt'
# ------------------------------------------------------------------------------
#                          Get Google Inception Model
# ------------------------------------------------------------------------------

# download the inception_v3 model at:
# https://github.com/tensorflow/models/tree/master/research/slim
utility.reset_graph()

X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
        X, num_classes=1001, is_training=False)
    predictions = end_points['Predictions']
    saver = tf.train.Saver()

# get imagenet classes
classes = []
with open('imagenet_class_names.txt', 'r') as r:
    for line in r.readlines():
        classes.append(re.split(r'n[0-9]+ ', line)[1].replace('\n', ''))

# ------------------------------------------------------------------------------
#                           Use it as a classifier
# ------------------------------------------------------------------------------

# get images
imgs = ['imgs/felix_small.jpg',
        'imgs/logan_small.jpg',
        'imgs/bebe_small.jpg',
        'imgs/donny_small.jpg',
        'imgs/hotdog_small.jpg']

imgs = [np.array(Image.open(img)).reshape((1, 299, 299, 3)) for img in imgs]
imgs = np.concatenate(imgs, axis=0)
imgs = imgs / float(255)  # normalize to be between 0 and 1 !!!

with tf.Session() as sess:
    saver.restore(sess, inception_path)
    predictions_val = predictions.eval(feed_dict={X: imgs})

# get top 5 categories

for i in xrange(imgs.shape[0]):
    top5 = np.argsort(predictions_val[i])[-5:]
    print predictions_val[i][top5]
    print np.array(classes)[top5]

''
# ------------------------------------------------------------------------------
#                           Re-train it on new task
# ------------------------------------------------------------------------------

utility.reset_graph()

# load shoes data
imgs_nike = [np.array(Image.open(img), dtype=np.float32).reshape((1, 299, 299, 3))
             for img in glob.glob('shoes_processed/nike*.jpg')]
imgs_nike = np.concatenate(imgs_nike, axis=0)

imgs_addidas = [np.array(Image.open(img), np.float32).reshape((1, 299, 299, 3))
                for img in glob.glob('shoes_processed/addidas*.jpg')]
imgs_addidas = np.concatenate(imgs_addidas, axis=0)

labels = np.r_[np.zeros(len(imgs_nike)), np.ones(len(imgs_addidas))]
imgs_all = np.concatenate([imgs_nike, imgs_addidas]) / float(255)

plt.subplot(1, 2, 1)
plt.imshow(imgs_nike[10])
plt.subplot(1, 2, 2)
plt.imshow(imgs_addidas[10])
plt.show()


# shuffle train / test #
n_train = 1000
ix = np.random.permutation(len(labels))
X_train, y_train = imgs_all[ix][:n_train], labels[ix][:n_train]
X_test, y_test = imgs_all[ix][n_train:], labels[ix][n_train:]

# reload inception model
X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(
        X, num_classes=1001, is_training=training)
    saver = tf.train.Saver()

# keep lower layers, but add a new output layer
n_outputs = 2
prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])
with tf.name_scope("output"):
    logits = tf.layers.dense(prelogits, n_outputs, name="logits")
    yhat = tf.nn.softmax(logits, name="yhat")

# create some variables for optimization / accuracy
y = tf.placeholder(tf.int32, shape=[None])
with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(0.001)
    tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
    training_op = optimizer.minimize(loss, var_list=tvars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

## Train the model ##
epochs = 30
batch_size = 4

acc = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    saver.restore(sess, inception_path)
    print 'Inception model restored'

    for epoch in xrange(epochs):

        kf = KFold(n_splits=len(y_train) // batch_size)
        folds = kf.split(y_train)
        k = 0
        for _, fold in folds:
            sess.run(training_op, feed_dict={X: X_train[fold],
                                             y: y_train[fold]})
            print '\t Iter: %d' % k
            k += 1
        test_ix = np.random.permutation(len(y_test))[:100]
        test_acc = accuracy.eval(feed_dict={X: X_test[ix], y: y_test[ix]})
        acc.append(test_acc)
        print 'Epoch: %d Test Accuracy: %f' % (epoch, test_acc)

    # create test predictions
    test_pred = sess.run(yhat, feed_dict={X: X_test})


plt.imshow(X_test[3])
plt.show()
