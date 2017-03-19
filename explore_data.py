import numpy as np
import pandas as pd

# ########## IMPORT DATA #############

df = pd.read_csv('data/train.csv')
unique_qids = np.unique(np.hstack((df['qid1'], df['qid2'])))

print('Total number of questions = %d' % unique_qids.shape[0])

# ########### EXTRACT FEATURES ########

from sklearn.feature_extraction import text # noqa
from sklearn.manifold import TSNE # noqa

print('Doing TFIDF')
tfidf = text.TfidfVectorizer(max_df=1)
X = tfidf.fit_transform(df['question1'])

print('Doing TSNE')
tsne = TSNE(n_components=2, init='pca', random_state=0)
Y = tsne.fit_transform(X[:1000].toarray())

# ########### Plot the data ###########

print('Plot')
import matplotlib.pyplot as plt # noqa
plt.plot(Y[:, 0], Y[:, 1], 'bo')
plt.show()
