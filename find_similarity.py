import numpy as np
import pandas as pd

from itertools import product

from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer

from sklearn.externals.joblib import Parallel, delayed

tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')


def find_best_sentence_match(words1, words2):
    """Find match between words of sentence 1 to words of sentence 2.

    Parameters
    ----------
    words1 : list of strings
        Words of the first sentence
    words2 : list of strings
        Words of the second sentence

    Returns
    -------
    sim : list of floats
        Similarity scores for each word in the first sentence
    """

    sim = list()
    for w1 in words1:
            best_score = list()
            for w2 in words2:
                best_score.append(_find_best_word_match(w1, w2))

            if np.nanmax(best_score):  # XXX: np.nanmax
                sim.append(np.nanmax(best_score))
            else:
                sim.append(np.nan)
    return np.array(sim)


def _path_similarity(s1, s2):
    sim = s1.path_similarity(s2)
    if sim is None:
        sim = np.nan
    return sim


def _find_best_word_match(w1, w2):
    syn1 = wn.synsets(w1)
    syn2 = wn.synsets(w2)

    if syn1 and syn2:
        return np.nanmax([_path_similarity(s1, s2) for (s1, s2)
                          in product(syn1, syn2)])
    else:
        return np.nan


def find_sentence_similarity(sent1, sent2):
    """Average the asymmetric match between sentence 1 and sentence 2."""

    words1 = tokenizer.tokenize(sent1)
    words2 = tokenizer.tokenize(sent2)

    sim_max = (np.nanmean(find_best_sentence_match(words1, words2)) +
               np.nanmean(find_best_sentence_match(words2, words1))) / 2
    return sim_max


if __name__ == '__main__':

    from sklearn.cross_validation import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import spearmanr

    n_samples = 400
    n_jobs = 4
    df = pd.read_csv('data/train.csv')

    # let's find similarity
    def func(series):
        sim = find_sentence_similarity(series['question1'],
                                       series['question2'])
        print('Question pair %d, %0.2f, duplicate=%d'
              % (series['id'], sim, series['is_duplicate']))
        return sim

    question1 = df['question1'].values[:n_samples]
    question2 = df['question2'].values[:n_samples]
    parallel_similarity = delayed(find_sentence_similarity)
    sim = Parallel(n_jobs=n_jobs, verbose=10)(
        parallel_similarity(df.ix[idx]['question1'], df.ix[idx]['question2'])
        for idx in range(n_samples))
    sim = np.array(sim)

    # let's compute correlation
    corr = spearmanr(sim, df['is_duplicate'][:n_samples].values)
    print(corr)

    # let's do classification
    X = sim[~np.isnan(sim)][:, None]
    y = df['is_duplicate'][:n_samples][~np.isnan(sim)]

    skf = StratifiedKFold(y, n_folds=5)
    clf = LogisticRegression()
    score = -cross_val_score(clf, X, y, scoring='neg_log_loss', cv=skf)
    print('Prediction accuracy is %f' % np.mean(score))
