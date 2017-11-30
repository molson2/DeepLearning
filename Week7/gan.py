import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


n_hidden_g = 100
n_hidden_d = 100
n_latent = 75
im_width = 28

eta_d = 0.001
eta_g = 0.001
batch_size = 128

outdir = 'big/'

tf.reset_default_graph()
tf.set_random_seed(123)
np.random.seed(123)

# ------------------------------------------------------------------------------
#                            Helper Funcs
# ------------------------------------------------------------------------------


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def sample(n, z_dim):
    return np.random.uniform(-1., 1., (n, z_dim))

# ------------------------------------------------------------------------------
#                            Load Data
# ------------------------------------------------------------------------------

mnist = input_data.read_data_sets('../../../Data/MNIST/', one_hot=True)
samp_ims, _ = mnist.train.next_batch(16)
plot(samp_ims)
plt.savefig('samp_mnist.png', bbox_inches='tight')

# ------------------------------------------------------------------------------
#                            Build the GAN
# ------------------------------------------------------------------------------

xavier_init = tf.contrib.layers.xavier_initializer()


def D(X, reuse=None):
    with tf.variable_scope('D', reuse=reuse):
        h1 = tf.layers.dense(X, n_hidden_d, activation=tf.nn.elu,
                             kernel_initializer=xavier_init)
        logit = tf.layers.dense(h1, im_width * im_width,
                                kernel_initializer=xavier_init)
        prob = tf.nn.sigmoid(logit)
    return prob, logit


def G(z):
    with tf.variable_scope('G'):
        h1 = tf.layers.dense(z, n_hidden_g, activation=tf.nn.elu,
                             kernel_initializer=xavier_init)
        h2 = tf.layers.dense(h1, n_hidden_g, activation=tf.nn.elu,
                             kernel_initializer=xavier_init)
        logit = tf.layers.dense(h2, im_width * im_width,
                                kernel_initializer=xavier_init)
        prob = tf.nn.sigmoid(logit)
    return prob

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, im_width * im_width))
z = tf.placeholder(tf.float32, shape=(None, n_latent))

sample_g = G(z)
p_real, logit_real = D(X)
p_fake, logit_fake = D(sample_g, reuse=True)

#loss_d = -tf.reduce_mean(tf.log(p_real) + tf.log(1. - p_fake))
#loss_g = -tf.reduce_mean(tf.log(p_fake))

loss_d = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logit_real, labels=tf.ones_like(logit_real)) +
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logit_fake, labels=tf.zeros_like(logit_fake)))

loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=logit_fake, labels=tf.ones_like(logit_fake)))


# ------------------------------------------------------------------------------
#                            Train the GAN
# ------------------------------------------------------------------------------

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

train_d = tf.train.AdamOptimizer(eta_d).minimize(loss_d, var_list=d_vars)
train_g = tf.train.AdamOptimizer(eta_g).minimize(loss_g, var_list=g_vars)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(outdir):
    os.makedirs(outdir)

fig_num = 0

for it in xrange(int(1e5)):
    if it % 1000 == 0:
        im_samps = sess.run(sample_g, feed_dict={z: sample(16, n_latent)})

        fig = plot(im_samps)
        plt.savefig(outdir + '{}.png'.format(str(fig_num).zfill(3)),
                    bbox_inches='tight')
        fig_num += 1
        plt.close(fig)

    X_batch, _ = mnist.train.next_batch(batch_size)

    _, loss_d_ = sess.run([train_d, loss_d],
                          feed_dict={X: X_batch, z: sample(batch_size, n_latent)})
    _, loss_g_ = sess.run([train_g, loss_g],
                          feed_dict={z: sample(batch_size, n_latent)})

    if it % 1000 == 0:
        print 'Iter: {} loss_d: {:.4} loss_g: {:.4}'.format(it, loss_d_, loss_g_)
