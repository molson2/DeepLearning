from gensim import utils
from gensim.models.doc2vec import LabeledSentence
import string
import numpy as np
import pandas as pd
import random
import glob
from collections import defaultdict
import re

SOURCES = ['New York Times', 'Fox News',
           'Reuters', 'National Review', 'Atlantic']
NEWS_PATH = '/Users/matthewolson/Documents/Data/News/'

# ------------------------------------------------------------------------------
#                             Helper Class
# ------------------------------------------------------------------------------

# https://github.com/linanqiu/word2vec-sentiments/blob/master/word2vec-sentiment.ipynb


class LabeledLineSentence(object):

    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return shuffled

# ------------------------------------------------------------------------------
#                               Read in News Sources
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    ## read in raw ##
    paths = glob.glob(NEWS_PATH + '*.csv')

    source_raw = defaultdict(list)
    for path in paths:
        df = pd.read_csv(path, header=0, encoding='utf-8')
        for source in SOURCES:
            source_raw[source].append(
                df.query('publication == @source').content.values)

    ## process ##
    for source in SOURCES:
        source_raw[source] = np.concatenate(source_raw[source])
        with open(source.replace(' ', '_') + '.txt', 'w') as f:
            for article in source_raw[source]:
                # kill unicode
                article = article.encode('unicode_escape')
                article = re.sub(r'(\\u[0-9A-Fa-f]+)', '', article)
                # kill \x
                article = re.sub(r'(\\x[0-9A-Fa-f]+)', '', article)
                # kill extra whitespace
                article = re.sub('\s+', ' ', article)
                # kill punctutation
                article = re.sub('[' + string.punctuation + ']', '', article)
                # remove references to publications
                article = re.sub(source, '', article)
                # make lower
                article = article.lower()
                # write
                f.write(article + '\n')
        print 'Done with %s' % source


''
# ------------------------------------------------------------------------------
#                                 Article Counts
# ------------------------------------------------------------------------------

# Breitbart           23781
# CNN                 11488
# New York Times       7803
# Business Insider     6757
# Atlantic              171

# New York Post          17493
# Atlantic                7008
# National Review         6203
# Talking Points Memo     5214
# Guardian                4873
# Buzzfeed News           4854
# Fox News                4354

# NPR                11992
# Washington Post    11114
# Reuters            10710
# Vox                 4947
# Guardian            3808
