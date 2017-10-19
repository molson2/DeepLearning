import nltk
import itertools
import numpy as np


class TextTokenizer(object):

    def __init__(self, raw_name, vocab_size):

        self.raw_name = raw_name
        self.start_token = 'START_TOKEN'
        self.unknown_token = 'UNKNOWN_TOKEN'
        self.end_token = 'END_TOKEN'
        self.vocab_size = vocab_size

    def tokenize(self):

        with open(self.raw_name) as f:
            txt_raw = f.read()
            txt_raw = txt_raw.lower().strip()

            # tokenize sentences
            tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            sentences = tokenizer.tokenize(txt_raw)
            tokenized_sentences = [[self.start_token] + nltk.word_tokenize(s) +
                                   [self.end_token] for s in sentences]

            # extract the vocab from most common vocab_size words
            freqs = nltk.FreqDist(itertools.chain(*tokenized_sentences))
            self.vocab = freqs.most_common(self.vocab_size - 1)

            self.index_to_word = [x[0] for x in self.vocab]
            self.index_to_word.append(self.unknown_token)
            self.word_to_index = dict([(w, i)
                                       for i, w, in enumerate(self.index_to_word)])

            # replace all words not in vocab with unknown_token
            for i, s in enumerate(tokenized_sentences):
                tokenized_sentences[i] = [w if w in self.word_to_index
                                          else self.unknown_token for w in s]

            # map everything to numpy arrays
            tokenized_data = np.asarray([[self.word_to_index[w] for w in s]
                                         for s in tokenized_sentences])
            return tokenized_data

tokenizer = TextTokenizer('temp.txt', 50)
tokenized_data = tokenizer.tokenize()
