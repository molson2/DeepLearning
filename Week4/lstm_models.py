import tensorflow as tf
import numpy as np
import pdb


class LSTMRegressor(object):

    def __init__(self, n_neurons=100, n_layers=1, n_steps=10,
                 save_path='my_ts.ckpt'):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.save_path = save_path
        self.is_trained = False

    def _make_batches(self, n_train, batch_size):
        '''
        create indices for minibatches
        '''
        ix = np.random.permutation(n_train)
        n_batches = np.ceil(float(n_train) / batch_size)
        batches = [ix[i * batch_size:(i + 1) * batch_size]
                   for i in range(int(n_batches))]
        return batches

    def _wrap(self, X):
        '''
        wrap series X into Toeplitz-like matrix; return X, y
        '''
        z = np.array([X[i:i + self.n_steps]
                      for i in range(X.shape[0] - self.n_steps + 1)])
        return z[: -1].reshape(-1, self.n_steps, 1), z[1:].reshape(-1, self.n_steps, 1)

    def fit(self, X_train, X_test, eta=0.001, n_epochs=100, batch_size=64):
        tf.reset_default_graph()
        ## Create the network ##
        X = tf.placeholder(tf.float32, [None, self.n_steps, 1])
        y = tf.placeholder(tf.float32, [None, self.n_steps, 1])
        lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons)
                      for layer in range(self.n_layers)]
        multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        lstm_outputs, states = tf.nn.dynamic_rnn(
            multi_cell, X, dtype=tf.float32)
        stacked_rnn_outputs = tf.reshape(lstm_outputs, [-1, self.n_neurons])
        he_init = tf.contrib.layers.variance_scaling_initializer()
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, 1,
                                          kernel_initializer=he_init)
        outputs = tf.reshape(stacked_outputs, [-1, self.n_steps, 1])
        loss = tf.reduce_mean(tf.square(outputs - y))

        ## Set the optimizer ##
        optimizer = tf.train.AdamOptimizer(learning_rate=eta)
        training_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        ## Gradient Descent ##
        X_train_, y_train_ = self._wrap(X_train)
        X_test_, y_test_ = self._wrap(X_test)
        n_train = X_train_.shape[0]

        # save the important variables so we can restore them for predictions
        self._X, self._y, self._outputs = X, y, outputs

        with tf.Session() as sess:
            init.run()
            for epoch in range(n_epochs):
                batches = self._make_batches(n_train, batch_size)
                for batch in batches:
                    sess.run(training_op, feed_dict={
                             X: X_train_[batch], y: y_train_[batch]})
                if epoch % 10 == 0:
                    train_accuracy = sess.run(
                        loss, feed_dict={X: X_train_, y: y_train_})
                    test_accuracy = sess.run(
                        loss, feed_dict={X: X_test_, y: y_test_})
                    print_str = 'Epoch: %d Train_MSE: %f Test_MSE: %f' % (
                        epoch, train_accuracy, test_accuracy)
                    print print_str
            saver.save(sess, self.save_path)
            self.is_trained = True

    def predict(self, x0, n_forc):
        '''
        starting at x0, predict n_steps forward
        @param x0 vector of last self.n_steps data points !!!
        '''
        assert len(x0) == self.n_steps
        if not self.is_trained:
            raise ValueError('Model is not yet trained!')
        x_test = x0.copy()
        yhat = np.zeros(n_forc)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            for i in xrange(n_forc):
                x_feed = x_test.reshape(-1, self.n_steps, 1)
                yhat[i] = sess.run(self._outputs, feed_dict={
                    self._X: x_feed})[0, -1, 0]
                x_test = np.append(x_test[1:], yhat[i])
        return yhat


# ------------------------------------------------------------------------------
#                               LSTM Lyrics
# ------------------------------------------------------------------------------


class LSTMLyrics(object):

    def __init__(self, raw_text, n_neurons=50, n_layers=1, n_steps=25,
                 save_path='my_lyrics.ckpt'):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.save_path = save_path
        self.is_trained = False
        self.raw_text = raw_text
        self.n = len(raw_text)
        self.vocab = list(set(raw_text))
        self.vocab_size = len(self.vocab)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.vocab)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        counts = np.array(map(lambda x: self.raw_text.count(x), self.vocab))
        self._freq = counts / (np.sum(counts) + 1e-5)
        print 'Model with vocab size: %d' % self.vocab_size
        print '\t Hidden neurons: %d, Hidden layers: %d' % (n_neurons, n_layers)

    def _softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    def next_batch(self, batch_size):
        '''
        Return next X, y batch of size batch_size
        (note: y[j] = x[j+1]; everything is one-hot coded)
        performs left to right sweep of self.raw_text
        X_batch = [n_batch x n_steps x vocab_size]
        '''
        X_batch = np.zeros((batch_size, self.n_steps, self.vocab_size))
        y_batch = np.zeros((batch_size, self.n_steps, self.vocab_size))
        for i in xrange(batch_size):

            # reset batch indices when they exceed text size
            if self._endix >= self.n:
                self._startix, self._endix = 0, self.n_steps + 1

            pattern = [self.char_to_ix[ch]
                       for ch in self.raw_text[self._startix:self._endix]]
            for j in xrange(self.n_steps):
                X_batch[i, j, pattern[j]] = 1
                y_batch[i, j, pattern[j + 1]] = 1

            # increment batch indices
            self._startix += self.n_steps
            self._endix += self.n_steps

        return X_batch, y_batch

    def fit(self, eta=0.001, n_epochs=100, batch_size=32, nchar=120):

        tf.reset_default_graph()

        X = tf.placeholder(tf.float32, [None, self.n_steps, self.vocab_size])
        y = tf.placeholder(tf.float32, [None, self.n_steps, self.vocab_size])

        lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=self.n_neurons)
                      for layer in range(self.n_layers)]
        multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        lstm_outputs, states = tf.nn.dynamic_rnn(
            multi_cell, X, dtype=tf.float32)

        stacked_rnn_outputs = tf.reshape(lstm_outputs, [-1, self.n_neurons])
        he_init = tf.contrib.layers.variance_scaling_initializer()
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, self.vocab_size,
                                          kernel_initializer=he_init)
        logits = tf.reshape(
            stacked_outputs, [-1, self.n_steps, self.vocab_size])

        loss_all = tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(loss_all)

        ## Set the optimizer ##
        optimizer = tf.train.AdamOptimizer(learning_rate=eta)
        training_op = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # save the important variables so we can restore them for predictions
        self._X, self._y, self._logits = X, y, logits
        self._startix, self._endix = 0, self.n_steps + 1

        n_batches = self.n // batch_size
        with tf.Session() as sess:
            init.run()
            for epoch in xrange(n_epochs):
                for batch in xrange(n_batches):
                    X_batch, y_batch = self.next_batch(batch_size)
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                    batch_loss = sess.run(
                        loss, feed_dict={X: X_batch, y: y_batch})
                    print '-' * 50
                    print 'Batch loss: %f' % batch_loss
                    print '-' * 50
                    print self.sample(nchar, sess)

            saver.save(sess, self.save_path)
            self.is_trained = True

    def sample(self, nchar, sess, x0=None):
        if x0 is None:
            x0 = np.random.choice(self.vocab, p=self._freq)
        x_feed = np.zeros((1, self.n_steps, self.vocab_size))
        x_feed[0][0][self.char_to_ix[x0]] = 1
        sentence = [x0]
        for k in xrange(nchar - 1):
            logits_ = sess.run(self._logits, feed_dict={self._X: x_feed})
            logits_next = logits_[0][min(k, self.n_steps - 1)]
            phat_next = self._softmax(logits_next)
            char_next = np.random.choice(self.vocab, p=phat_next)
            sentence.append(char_next)
            if k < self.n_steps:
                x_feed[0][k][self.char_to_ix[char_next]]
            else:
                x_feed[0][:-1] = x_feed[0][1:]
                x_feed[0][-1][self.char_to_ix[char_next]] = 1
        return ''.join(sentence)
