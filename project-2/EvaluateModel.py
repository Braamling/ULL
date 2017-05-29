"""
Unsupervised Language Learning (ULL) 2017 - project 2

### DataModel
This model is a helper class for loading and using the preprocessed dataset

### Authors
Jorn Peters & Bram van den Akker

"""

import pickle
import operator
import random

import numpy as np

from collections import Counter

class EvaluateModel():
    def __init__(self, config):
        self.config = config

        self.id2word = pickle.load(open(self.config.id2word, 'rb'))
        self.id2vector = np.asarray(pickle.load(open(self.config.id2vector, 'rb')))
        self.id2rel = pickle.load(open(self.config.id2rel, 'rb'))
        self.test_pairs = pickle.load(open(self.config.test_pairs, 'rb'))
        self.train_pairs = pickle.load(open(self.config.train_pairs, 'rb'))

        counts = Counter([r for v, n, r in self.train_pairs])
        s_list = sorted(counts.items(), key=lambda x: -x[1])

        # Keep the keys of the N most frequent relationships
        self.valid_rels = [idx for idx, freq in s_list[:self.config.rel_cutoff]]
        # map relations to 0 to N indices
        self.rel2idx = {rel: idx for idx, rel in enumerate(self.valid_rels)}

        self.batch_pointer = 0

    def filter_relationships(self):
        # Remove all invalid relationships
        self.pairs = [(v, n, r) for v, n, r in self.pairs if r in self.valid_rels]

        # Create lists of the indeces of the verb vectors ,noun vectors and relationids
        verb_idx, noun_idx, rel_idx = map(list, zip(*self.pairs))

        self.verb_counts = Counter(verb_idx)
        self.noun_counts = Counter(noun_idx)
        self.rel_counts = Counter(rel_idx)

    def convert_pairs_to_vectors(self, pairs):
        verb_idx, noun_idx, rel_idx = \
                map(list, zip(*pairs))

        # Extract the verb and noun vectors
        verb_vecs = self.id2vector[verb_idx]
        noun_vecs = self.id2vector[noun_idx]

        # Create NxM relationship multihots
        rel_hots = np.zeros((len(rel_idx), self.config.rel_cutoff))

        # Create test multihot
        for i, rel in enumerate(rel_idx):
            rel_hots[i, self.rel2idx[rel]] = 1

        return verb_vecs, noun_vecs, rel_hots
