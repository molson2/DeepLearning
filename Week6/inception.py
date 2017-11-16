import sys
sys.path.insert(0, '../Week5/')
import utility
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

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
