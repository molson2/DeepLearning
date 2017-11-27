import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import sys
sys.path.append('../Week5/')
import utility

# ------------------------------------------------------------------------------
#                              Load in Data
# ------------------------------------------------------------------------------

d_path = '/Users/matthewolson/Documents/Data/Fashion/'
X_data, y_data = utility.read_fashion(range(10), 'training', d_path)

# reshape X_data into (60000, 28, 28, 1)

# ------------------------------------------------------------------------------
#                                Setup
# ------------------------------------------------------------------------------

n_latent = 50
n_hidden_g1 = 128
n_hidden_g2 = 128
n_conv_d = 16
n_hidden_d = 128
n_input = 28


def G(z):
    '''
    Takes in latent z and outputs 28x28 probability image
    '''
    with tf.variable_scope('G'):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        hidden1 = tf.layers.dense(z, n_hidden_g1,
                                  kernel_initializer=he_init,
                                  activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden_g2,
                                  kernel_initializer=he_init,
                                  activation=tf.nn.relu)
        im_prob = tf.layers.dense(hidden2, n_input * n_input,
                                  kernel_initializer=he_init,
                                  activation=tf.nn.sigmoid)
        im_prob = tf.reshape(im_prob, shape=(-1, n_input, n_input, 1))
    return im_prob


def D(x, reuse=None):
    '''
    Takes in (batch_size, 28, 28, 1) set of images, outputs prob
    '''
    with tf.variable_scope('D', reuse=reuse):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        conv = tf.layers.conv2d(X, filters=n_conv_d,
                                kernel_size=3, strides=2,
                                padding='SAME',
                                activation=tf.nn.relu)
        pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
        pool_flat = tf.reshape(pool, shape=(-1, n_conv_d * 7 * 7))
        hidden = tf.layers.dense(pool_flat, n_hidden_d, activation=tf.nn.relu)
        logits = tf.layers.dense(hidden, 2)
        y_prob = tf.nn.softmax(logits)
    return y_prob


X = tf.placeholder(tf.float32, shape=(None, n_input, n_input, 1))
z = tf.placeholder(tf.float32, shape=(None, n_latent))

sample_g = G(z)
prob_real = D(X)
prob_fake = D(sample_g, reuse=True)

loss_d = -tf.reduce_mean(tf.log(prob_real) + tf.log(1. - prob_fake))
loss_g = -tf.reduce_mean(tf.log(prob_fake))

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

training_op_d = tf.train.AdamOptimizer(0.001).minimize(loss_d, var_list=d_vars)
training_op_g = tf.train.AdamOptimizer(0.001).minimize(loss_g, var_list=g_vars)


# ------------------------------------------------------------------------------
#                                 Train GAN
# ------------------------------------------------------------------------------

n_epochs = 100
batch_size = 128
n_samps = 16

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()

    for epoch in xrange(n_epochs):
        kf = KFold(n_splits=len(images) // batch_size, shuffle=True)
        batches = kf.split(images)
        for _, batch in batches:

            batch_len = len(batch)

            ## update D ##
            Z_sample = np.random.random((batch_len, n_latent))
            _, loss_d_ = sess.run([training_op_d, loss_d],
                                  feed_dict={X: X[batch], z: z_sample})

            ## update G ##
            Z_sample = np.random.random((batch_len, n_latent))

            _, loss_g_ = sess.run([training_op_g, loss_g],
                                  feed_dict={z: z_sample})
        print 'Epoch: %d loss_d: %f  loss_g: %f' % (epoch, loss_d_, loss_g_)
        save_path = saver.save(sess, 'fashion_gan.ckpt')

        # sample some images
        z_sample = np.random.random((n_samps, n_latent))
        samp_ims = sess.run(prob_fake, feed_dict={z: z_sample})
