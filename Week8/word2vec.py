from word2vec_preprocess import TextTokenizer
import tensorflow as tf
import numpy as np

VOCAB_SIZE = 7500
EMBEDDING_DIM = 100
# ------------------------------------------------------------------------------
#                         Load the Data and Tokenize Sentences
# ------------------------------------------------------------------------------

text_token = TextTokenizer('Fox_Small.txt', VOCAB_SIZE)
text_token.tokenize()

# example output
print ' '.join([text_token.index_to_word[w]
                for w in text_token.tokenized_sentences[0]])

context, target = text_token.next_batch(window=2)
[text_token.index_to_word[w] for w in target]
[text_token.index_to_word[w] for w in context]


# ------------------------------------------------------------------------------
#                         Build Word2Vec Model
# ------------------------------------------------------------------------------

learning_rate = 0.001
tf.reset_default_graph()


target_labels = tf.placeholder(tf.int64, shape=[None, 1])
context_labels = tf.placeholder(tf.int64, shape=[None])
embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_DIM], -1, 1))
embed = tf.nn.embedding_lookup(embeddings, context_labels)

weights = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBEDDING_DIM],
                                          stddev=1.0 / np.sqrt(EMBEDDING_DIM)))
biases = tf.Variable(tf.zeros([VOCAB_SIZE]))

loss = tf.reduce_mean(tf.nn.sampled_softmax(weights=weights,
                                            biases=biases,
                                            labels=target_labels,
                                            inputs=embed,
                                            num_sampled=5,
                                            num_classes=VOCAB_SIZE))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# ------------------------------------------------------------------------------
#                               Train
# ------------------------------------------------------------------------------
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

max_iter = int(1e5)
for k in xrange(max_iter):
    x_batch, y_batch = text_token.next_batch(window=2)
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch).reshape(-1, 1)
    sess.run(training_op, feed_dict={context_labels: x_batch,
                                     target_labels: y_batch})
    if k % 1000 == 0:
        print 'Iter: %d' % k


''
# ------------------------------------------------------------------------------
#                               Some Results
# ------------------------------------------------------------------------------

sess.close()
