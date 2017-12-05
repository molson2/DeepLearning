from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
import random
import glob
from news_preprocess import LabeledLineSentence

# ------------------------------------------------------------------------------
#                      Read in Articles (see news_preprocess.py)
# ------------------------------------------------------------------------------

sources = {'Fox_News.txt': 'Fox_News',
           'National_Review.txt': 'National_Review',
           'New_York_Times.txt': 'New_York_Times',
           'Reuters.txt': 'Reuters'}

sentences = LabeledLineSentence(sources)

# ------------------------------------------------------------------------------
#                          Train Doc2Vec
# ------------------------------------------------------------------------------

model = Doc2Vec(min_count=10, window=10, size=100,
                sample=1e-4, negative=5, workers=4)
model.build_vocab(sentences.to_array())
model.train(sentences.sentences_perm(),
            total_examples=model.corpus_count, epochs=5)

# ------------------------------------------------------------------------------
#                      Run Classifier on Doc Features
# ------------------------------------------------------------------------------
