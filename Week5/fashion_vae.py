import tensorflow as tf
import numpy as np
import utility
import matplotlib.pyplot as plt
import sys
from functools import partial
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
EPS = 1e-10

# ------------------------------------------------------------------------------
#                         Load Data
# ------------------------------------------------------------------------------

path = '/Users/matthewolson/Documents/Data/Fashion'
images, labels = utility.read_fashion(range(10), dataset='training', path=path)

# make some plots


# ------------------------------------------------------------------------------
#                       Build the VAE
# ------------------------------------------------------------------------------

utility.reset_graph()

n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 50
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

he_init = tf.contrib.layers.variance_scaling_initializer()
dense_layer = partial(tf.layers.dense, activation=tf.nn.elu,
                      kernel_initializer=he_init)

X = tf.placeholder(tf.float32, (None, n_inputs))
hidden1 = dense_layer(X, n_hidden1)
hidden2 = dense_layer(hidden1, n_hidden2)
hidden3_mean = dense_layer(hidden2, n_hidden3, activation=None)
hidden3_sigma = dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
hidden3 = hidden3_mean + hidden3_sigma * noise
hidden4 = dense_layer(hidden3, n_hidden4)
hidden5 = dense_layer(hidden4, n_hidden5)
outputs = dense_layer(hidden5, n_outputs, activation=None)

reconstruction_loss = tf.reduce_sum(tf.square(X - outputs))

latent_loss = 0.5 * tf.reduce_sum(tf.square(hidden3_sigma) + tf.square(
    hidden3_mean) - 1 - tf.log(EPS + tf.square(hidden3_sigma)))

loss = reconstruction_loss + latent_loss

# ------------------------------------------------------------------------------
#                        Train the Model
# ------------------------------------------------------------------------------

learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

sc = StandardScaler()
sc.fit(images)

n_epochs = 50
batch_size = 512
with tf.Session() as sess:
    init.run()
    for epoch in xrange(n_epochs):
        kf = KFold(n_splits=len(images) // batch_size, shuffle=False)
        for _, batch in kf.split(images):
            X_batch = sc.transform(images[batch])
            sess.run(training_op, feed_dict={X: X_batch})
        loss_val, recon_loss_val = sess.run(
            [loss, reconstruction_loss], feed_dict={X: X_batch})
        out_str = 'Epoch %d: Total loss: %f Recon loss %f' % (
            epoch, loss_val, recon_loss_val)
        print out_str
        saver.save(sess, "./fashion_vae.ckpt")


# ------------------------------------------------------------------------------
#                           Generate New Digits
# ------------------------------------------------------------------------------

n_digits = 20

with tf.Session() as sess:
    saver.restore(sess, 'fashion_vae.ckpt')
    codings = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings})
    outputs_val = sc.inverse_transform(outputs_val)

utility.plot_multiple_images(outputs_val.reshape(-1, 28, 28), 4, 5)
plt.savefig('generated_digits.png', format='png', dpi=300)
plt.show()

# ------------------------------------------------------------------------------
#                           Compress / Decode Digits
# ------------------------------------------------------------------------------

img = images[np.random.randint(0, len(images), 1)]
img_scaled = sc.transform(img)

with tf.Session() as sess:
    saver.restore(sess, "fashion_vae.ckpt")
    codings = hidden3.eval(feed_dict={X: img_scaled})

with tf.Session() as sess:
    saver.restore(sess, "fashion_vae.ckpt")
    outputs_val = outputs.eval(feed_dict={hidden3: codings})
    img_recon = sc.inverse_transform(outputs_val)

plt.subplot(1, 2, 1)
plt.imshow(img_recon.reshape((28, 28)), cmap='gray', interpolation='bessel')
plt.subplot(1, 2, 2)
plt.imshow(img.reshape((28, 28)), cmap='gray', interpolation='bessel')
plt.savefig('compressed.png', format='png', dpi=300)
plt.show()
