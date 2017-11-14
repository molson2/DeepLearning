#  https://github.com/hwalsuklee/tensorflow-mnist-VAE/blob/master/vae.py
import numpy as np
import utility
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
from sklearn.model_selection import KFold
EPS = 1e-10
import pdb

# ------------------------------------------------------------------------------
#                         Load Data
# ------------------------------------------------------------------------------

path = '/Users/matthewolson/Documents/Data/Fashion'
images, labels = utility.read_fashion(range(10), dataset='training', path=path)


# rescale images to be between 0 / 1
images = images / float(images.max())

# make some plots
ix = np.random.permutation(len(images))[:20]
utility.plot_multiple_images(images[ix].reshape(-1, 28, 28), 4, 5)
plt.savefig('data_images.png', format='png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
#                            Build VAE Model
# ------------------------------------------------------------------------------


def gaussian_encoder(X, n_hidden, n_latent):
    with tf.variable_scope('guassian_encoder'):
        he_init = tf.contrib.layers.variance_scaling_initializer()
        hidden1 = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                                  kernel_initializer=he_init)
        hidden2 = tf.layers.dense(
            hidden1, n_hidden, activation=tf.nn.tanh, kernel_initializer=he_init)
        gaussian_params = tf.layers.dense(hidden2, n_latent * 2,
                                          kernel_initializer=he_init)
        mean = gaussian_params[:, :n_latent]
        stddev = EPS + tf.nn.softplus(gaussian_params[:, n_latent:])
    return mean, stddev


def bernoulli_decoder(Z, n_hidden, n_output):
    with tf.variable_scope('bernoulli_decoder'):
        #        pdb.set_trace()
        he_init = tf.contrib.layers.variance_scaling_initializer()
        hidden1 = tf.layers.dense(Z, n_hidden, activation=tf.nn.tanh,
                                  kernel_initializer=he_init)
        hidden2 = tf.layers.dense(hidden1, n_hidden, activation=tf.nn.elu,
                                  kernel_initializer=he_init)
        p = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid,
                            kernel_initializer=he_init)
    return p


def autoencoder(X, n_input, n_hidden, n_latent):
    #    pdb.set_trace()
    # encoding
    mu, sigma = gaussian_encoder(X, n_hidden, n_latent)

    # reparametrization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0.0, 1)

    # decoding
    p = bernoulli_decoder(z, n_hidden, n_input)

    # loss
    likelihood = tf.reduce_sum(X * tf.log(p) + (1 - X) * tf.log(1 - p), 1)
    kl_divergence = 0.5 * \
        tf.reduce_sum(tf.square(mu) + tf.square(sigma) -
                      tf.log(EPS + tf.square(sigma)) - 1, 1)
    likelihood = tf.reduce_mean(likelihood)
    kl_divergence = tf.reduce_mean(kl_divergence)

    loss = kl_divergence - likelihood
    return p, z, loss, -likelihood, kl_divergence

# ------------------------------------------------------------------------------
#                              Train Model
# ------------------------------------------------------------------------------

n_hidden = 500
n_inputs = 28 * 28
n_latent = 50

n_epochs = 500
batch_size = 128
learning_rate = 0.001

utility.reset_graph()
X = tf.placeholder(tf.float32, shape=[None, n_inputs], name='img')
p, z, loss, neg_likelihood, kl_divergence = autoencoder(
    X, n_inputs, n_hidden, n_latent)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)

    for epoch in xrange(n_epochs):
        kf = KFold(n_splits=len(images) // batch_size, shuffle=True)
        batches = kf.split(images)
        for _, batch in batches:
            sess.run(training_op, feed_dict={X: images[batch]})
        loss_total, loss_lhood, loss_kl = sess.run([loss, neg_likelihood,
                                                    kl_divergence],
                                                   feed_dict={X: images})
        print("Epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (
            epoch, loss_total, loss_lhood, loss_kl))
        saver.save(sess, 'fashion_vae_50.ckpt')

# ------------------------------------------------------------------------------
#                           Show Distribution of Labelled Data
# ------------------------------------------------------------------------------

# pass z through with X, label by color
saver = tf.train.Saver()
ix = np.random.permutation(len(images))[:1000]
with tf.Session() as sess:
    saver.restore(sess, 'fashion_vae_2.ckpt')
    z_vals = sess.run(z, feed_dict={X: images[ix]})

labels_set = list(set(labels[ix]))
for i in xrange(len(labels_set)):
    clothing_ix = labels[ix] == labels_set[i]
    plt.scatter(z_vals[clothing_ix, 0], z_vals[clothing_ix, 1], alpha=0.9)
plt.savefig('latent_images.png', format='png', dpi=300)
plt.show()


# ------------------------------------------------------------------------------
#                          Generate Examples
# ------------------------------------------------------------------------------

# pass p through with X

n_images = 20
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    saver.restore(sess, 'fashion_vae_50.ckpt')
    img = sess.run(p, feed_dict={z: np.random.randn(20, 50)})

utility.plot_multiple_images(img.reshape(-1, 28, 28), 4, 5)
plt.savefig('random_images.png', format='png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
#                        JUNK
# ------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()
sc.fit(images)
images_ = sc.fit_transform(images)
pca = PCA(n_components=2)
pca.fit(images_)

ix = np.random.permutation(len(images))[:3000]
U = pca.fit_transform(images_)
U = U[ix]

labels_set = list(set(labels[ix]))
for i in xrange(len(labels_set)):
    clothing_ix = labels[ix] == labels_set[i]
    plt.scatter(U[clothing_ix, 0], U[clothing_ix, 1], alpha=0.9)
plt.savefig('pca_images.png', format='png', dpi=300)
plt.show()
