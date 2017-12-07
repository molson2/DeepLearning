# batch: consist of a number of words -> all context pairs (so sent imp!)
# output (int_x, int_y) pairs
# re-populate pool when it runs out

import nltk
import itertools
import numpy as np
import re
from random import shuffle
import pandas as pd


class TextTokenizer(object):

    def __init__(self, raw_name, vocab_size):

        self.raw_name = raw_name
        self.unknown_token = 'UNKNOWN'
        self.vocab_size = vocab_size - 1

    def tokenize(self):

        with open(self.raw_name) as f:
            txt_raw = unicode(f.read(), 'utf-8')
            txt_raw = txt_raw.lower().strip()
            txt_raw = re.sub(r'[,;:\()@#]', '', txt_raw)

            # tokenize sentences
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = tokenizer.tokenize(txt_raw)
            tokenized_sentences = [nltk.word_tokenize(s) for s in sentences]

            # extract the vocab from most common vocab_size words
            freqs = nltk.FreqDist(itertools.chain(*tokenized_sentences))
            self.vocab = freqs.most_common(self.vocab_size - 1)

            self.index_to_word = [x[0] for x in self.vocab]
            self.index_to_word.append(self.unknown_token)
            self.word_to_index = dict([(w, i)
                                       for i, w, in enumerate(self.index_to_word)])

            # replace all words not in vocab with unknown_token
            for i, s in enumerate(tokenized_sentences):
                tokenized_sentences[i] = [self.word_to_index[w] if w in self.word_to_index
                                          else len(self.index_to_word) - 1 for w in s[:-1]]
        self.tokenized_sentences = filter(
            lambda x: len(x) > 5, tokenized_sentences)
        self.pool = range(len(self.tokenized_sentences))
        shuffle(self.pool)

    def next_batch(self, batch_size=1, window=2):
        # sample batch_size sentences
        if self.pool is None:
            raise Exception('Must run tokenize first')

        # repop pool
        if len(self.pool) < batch_size:
            self.pool = range(len(self.tokenized_sentences))
            shuffle(self.pool)
            print 'Done with Epoch'

        self.pool = self.pool[batch_size:]
        sentences = self.pool[:batch_size]

        target = []
        context = []
        for k in sentences:
            s = self.tokenized_sentences[k]
            for pos, w in enumerate(s):
                for nbw in s[max(pos - window, 0): min(pos + window, len(s)) + 1]:
                    if nbw != w:
                        target.append(w)
                        context.append(nbw)
        return context, target


''
# ------------------------------------------------------------------------------
#                               Dump News Article
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    df = pd.read_csv('/Users/matthewolson/Documents/Data/News/articles2.csv',
                     header=0, encoding='utf-8')
    content = df.query('publication == "Fox News"').content.values

    with open('Fox_Small.txt', 'w') as f:
        for article in content:
            article = article.encode('ascii', 'ignore')
            if article.find('Trump') > 0:
                f.write(article)
