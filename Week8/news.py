from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np
import random
import glob
from news_preprocess import LabeledLineSentence

n_epochs = 50
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
            total_examples=model.corpus_count, epochs=n_epochs)

# ------------------------------------------------------------------------------
#                      Run Classifier on Doc Features
# ------------------------------------------------------------------------------


## encode in numpy arrays / create train and test sets ##

X_train = []
y_train = []
X_test = []
y_test = []
test_keep = 1000
k = 0

for publication in sources.values():
    publication = publication.replace(' ', '_')
    tags = filter(lambda x: publication in x, model.docvecs.doctags.keys())
    random.shuffle(tags)
    for tag in tags[test_keep:]:
        X_train.append(model.docvecs[tag])
        y_train.append(k)
    for tag in tags[:test_keep]:
        X_test.append(model.docvecs[tag])
        y_test.append(k)
    k += 1

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

#sims = d2v_model.docvecs.most_similar(docvec)

## fit classifier ##
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


rf = RandomForestClassifier(n_estimators=200, verbose=3)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


confusion_matrix(y_test, y_pred)
