import re
RM_MAX = 1000000

# ------------------------------------------------------------------------------
#                      Read in / process lyrics
# ------------------------------------------------------------------------------

with open('bible.txt', 'r') as f:
    lyrics = f.read()

# remove titles, make everything lowercase
lyrics = re.sub(r'##[A-Z]+##', '', lyrics, RM_MAX)
lyrics = re.sub(r'\x80|\x93|\x94|\x98|\x99|\x9c|\x9d|\xa6|\xad|\xaf|\xc2|\xc3|\xe2|\xa9|\xa8|\xa1|\xa0|\xa7|\xef|\xbb|\xbf|-',
                '', lyrics, RM_MAX)
lyrics = re.sub(r'\$|#|&|\*|\{|\}|/|;|\)|\(|', '', lyrics, RM_MAX)
lyrics = lyrics.lower()

vocab = list(set(lyrics))
counts = map(lambda x: lyrics.count(x), vocab)
zip(vocab, counts)

# ------------------------------------------------------------------------------
#                         Train the Model
# ------------------------------------------------------------------------------


import lstm_models

lyric_gen = lstm_models.LSTMLyrics(
    lyrics, n_neurons=512, n_layers=2, n_steps=50, save_path='bible.ckpt')
lyric_gen.fit(n_epochs=100, batch_size=128, eta=0.001, nchar=200)
