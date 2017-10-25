import re

# ------------------------------------------------------------------------------
#                  Read in / process lyrics
# ------------------------------------------------------------------------------

with open('swift.csv', 'r') as f:
    lyrics = f.read()

# remove titles, make everything lowercase
lyrics = re.sub(r'##[A-Z]+##', '', lyrics, 1000)
lyrics = re.sub(r'\x80|\x93|\x94|\x98|\x99|\x9c|\x9d|\xa6|\xad|\xaf|\xc2|\xc3|\xe2',
                '', lyrics, 10000)
lyrics = re.sub(r'#|&|\*|[0-9]|\{|\}|/|;', '', lyrics, 10000)
lyrics = lyrics.lower()

vocab = list(set(lyrics))
counts = map(lambda x: lyrics.count(x), vocab)
zip(vocab, counts)

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import lstm_models

reload(lstm_models)
swift_gen = lstm_models.LSTMLyrics(lyrics, n_neurons=50, n_layers=1)
swift_gen.fit(n_epochs=100, batch_size=2048, eta=0.1)

# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------
