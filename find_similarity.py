import numpy as np
import pandas as pd

from itertools import product

from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer

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

            if max(best_score):  # XXX: np.nanmax
                sim.append(max(best_score))
            else:
                sim.append(np.nan)
    return np.array(sim)


def _find_best_word_match(w1, w2):
    syn1 = wn.synsets(w1)
    syn2 = wn.synsets(w2)

    if syn1 and syn2:
        return max(s1.path_similarity(s2) for (s1, s2)
                   in product(syn1, syn2))
    else:
        return None


def find_sentence_similarity(sent1, sent2):
    """Average the asymmetric match between sentence 1 and sentence 2."""

    words1 = tokenizer.tokenize(sent1)
    words2 = tokenizer.tokenize(sent2)

    sim_max = (np.nanmean(find_best_sentence_match(words1, words2)) +
               np.nanmean(find_best_sentence_match(words2, words1))) / 2
    return sim_max


if __name__ == '__main__':

    df = pd.read_csv('data/train.csv')

    def func(series):
        sim = find_sentence_similarity(series['question1'].decode('utf-8'),
                                       series['question2'].decode('utf-8'))
        print('Question pair %d, %0.2f, duplicate=%d'
              % (series['id'], sim, series['is_duplicate']))

    df.head(20).apply(func, axis=1)
