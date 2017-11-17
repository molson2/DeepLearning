import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# get image and convert to gray scale
img = np.array(Image.open('imgs/donny.jpg', 'r'))
height, width = img.shape[:2]
img_gray = img.mean(axis=2).astype(np.float32)
img = img_gray.reshape(1, height, width, 1)

plt.imshow(np.squeeze(img[0, :, :, :]), cmap='gray')
plt.show()

# ------------------------------------------------------------------------------
#                          Convolutional Layers
# ------------------------------------------------------------------------------


# vertical / horizonal filter
fmap = np.zeros(shape=(51, 51, 1, 2), dtype=np.float32)
fmap[:, 24:26, 0, 0] = 1
fmap[24:26, :, 0, 1] = 1

plt.subplot(1, 2, 1)
# vertical out
plt.title('Vertical Filter')
plt.axis('off')
plt.imshow(fmap[:, :, 0, 0], cmap='gray')

# horizontal out
plt.subplot(1, 2, 2)
plt.title('Horizontal Filter')
plt.axis('off')
plt.imshow(fmap[:, :, 0, 1], cmap='gray')
plt.savefig('slides/verthorz_filter.png')
plt.show()

tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
feature_maps = tf.constant(fmap)
convolution = tf.nn.conv2d(X, feature_maps, strides=[
                           1, 1, 1, 1], padding="SAME")

# calculate the convolution map
with tf.Session() as sess:
    output = convolution.eval(feed_dict={X: img})

plt.subplot(1, 3, 1)
# vertical out
plt.title('Actual Image')
plt.axis('off')
plt.imshow(img[0, :, :, 0], cmap='gray')

plt.subplot(1, 3, 2)
# vertical out
plt.title('Vertical Filter')
plt.axis('off')
plt.imshow(output[0, :, :, 0], cmap='gray')

# horizontal out
plt.subplot(1, 3, 3)
plt.title('Horizontal Filter')
plt.axis('off')
plt.imshow(output[0, :, :, 1], cmap='gray')
plt.savefig('slides/donny_filter.png')
plt.show()


# ------------------------------------------------------------------------------
#                               Max-Pooling
# ------------------------------------------------------------------------------

X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
max_pool = tf.nn.max_pool(X, ksize=[1, 10, 10, 1], strides=[
                          1, 2, 2, 1], padding="VALID")
with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: img})


plt.subplot(1, 2, 1)
# vertical out
plt.title('Actual Image')
plt.axis('off')
plt.imshow(img[0, :, :, 0], cmap='gray')

# max pooling
plt.subplot(1, 2, 2)
plt.axis('off')
plt.title('Max Pooling')
plt.imshow(output[0, :, :, 0].astype(np.uint8), cmap='gray')
plt.savefig('slides/donny_maxpool.png')
plt.show()
