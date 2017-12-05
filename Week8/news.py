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
           'Reuters.txt': 'Reuters',
           'Atlantic.txt': 'Atlantic'}

sentences = LabeledLineSentence(sources)

# ------------------------------------------------------------------------------
#                          Train Doc2Vec
# ------------------------------------------------------------------------------

model = Doc2Vec(min_count=5, window=10, size=100,
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

# sims = d2v_model.docvecs.most_similar(docvec)

## fit classifier ##
import glmnet
from sklearn.metrics import confusion_matrix


glm = glmnet.LogitNet()
glm.fit(X_train, y_train)
glm.score(X_test, y_test)
y_pred = glm.predict(X_test)

confusion_matrix(y_test, y_pred)

## fit classifier subset ##
subset = [1, 3]
ix_train = np.isin(y_train, subset)
ix_test = np.isin(y_test, subset)

glm = glmnet.LogitNet()
glm.fit(X_train[ix_train], y_train[ix_train])
glm.score(X_test[ix_test], y_test[ix_test])
y_pred = glm.predict(X_test[ix_test])

confusion_matrix(y_test[ix_test], y_pred)

# ------------------------------------------------------------------------------
#                              Visualize Space
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
sns.set(style="ticks", color_codes=True)

pca = PCA(n_components=2)
x_scaled = StandardScaler().fit_transform(X_train)
pca.fit(x_scaled)
ix = np.random.choice(len(X_train), 700)
pca_scores = pca.fit_transform(x_scaled[ix])

plt_df = pd.DataFrame({'x1': pca_scores[:, 0],
                       'x2': pca_scores[:, 1],
                       'label': [sources.values()[k] for k in y_train[ix]]})
g = sns.FacetGrid(plt_df, col='label', col_wrap=3)
g = g.map(plt.scatter, 'x1', 'x2', edgecolor="w")
plt.show()
