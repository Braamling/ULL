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

from utils import H5PYDatasetCreator
from collections import Counter

class DataModel():
    def __init__(self, config):
        self.config = config

        self.pairs = pickle.load(open(self.config.pairs, 'rb'))
        self.id2word = pickle.load(open(self.config.id2word, 'rb'))
        self.id2vector = np.asarray(pickle.load(open(self.config.id2vector, 'rb')))
        self.id2rel = pickle.load(open(self.config.id2rel, 'rb'))

        counts = Counter([r for v, n, r in self.pairs])
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

    def prepare_for_hsf5(self):
        test_verb_idx, test_noun_idx, test_rel_idx = \
                map(list, zip(*self.test_pairs))

        train_verb_idx, train_noun_idx, train_rel_idx = \
                map(list, zip(*self.train_pairs))

        # Extract the verb and noun vectors
        self.test_verb_vecs = self.id2vector[test_verb_idx]
        self.train_verb_vecs = self.id2vector[train_verb_idx]

        self.test_noun_vecs = self.id2vector[test_noun_idx]
        self.train_noun_vecs = self.id2vector[train_noun_idx]

        # Create NxM relationship multihots
        self.test_rel_hots = np.zeros((len(test_rel_idx), self.config.rel_cutoff))
        self.train_rel_hots = np.zeros((len(train_rel_idx), self.config.rel_cutoff))

        # Create test multihot
        for i, rel in enumerate(test_rel_idx):
            self.test_rel_hots[i, self.rel2idx[rel]] = 1

        # Create test multihot
        for i, rel in enumerate(train_rel_idx):
            self.train_rel_hots[i, self.rel2idx[rel]] = 1

    def split_test_train(self, output_size=3000, min_freq=30, max_freq=3000):
        self.filter_relationships()

        unique_pairs = list(set(self.pairs))
        self.test_pairs = []
        
        # pairs index to speedup the sampled pairs lookup
        pairs_index = {}
        while len(self.test_pairs) < output_size:
            sampled_pair, idx = self.sample_pair(unique_pairs)

            counts = [self.verb_counts[sampled_pair[0]],
                      self.noun_counts[sampled_pair[1]],
                      self.rel_counts[sampled_pair[2]]]

            if all(min_freq < i and max_freq > i for i in counts):
                self.test_pairs.append(sampled_pair)
                pairs_index[sampled_pair] = 0
                del unique_pairs[idx]

                self.verb_counts[sampled_pair[0]] -= 1
                self.noun_counts[sampled_pair[1]] -= 1
                self.rel_counts[sampled_pair[2]] -= 1

        self.train_pairs = [pair for pair in self.pairs if \
                            pair not in pairs_index]

    def sample_pair(self, pairs):
        idx = random.randint(0, len(pairs) - 1)

        return pairs[idx], idx    

    def store_H5PY(self):
        # Create all the datasets and split train/test
        self.prepare_for_hsf5()

        creator = H5PYDatasetCreator('/home/jorn/Desktop/outfile.test')

        creator.add_split('train', len(self.train_rel_hots))
        creator.add_split('test', len(self.test_rel_hots))
        creator.add_source('noun_vec', self.config.vec_shape, np.float32)
        creator.add_source('verb_vec', self.config.vec_shape, np.float32)
        creator.add_source('rel', (self.config.rel_cutoff,), np.float32)
        creator.add_source('idx', (1,), np.int)

        # Add the train data to H5PY
        train_data = zip(self.train_verb_vecs, self.train_noun_vecs,
                         self.train_rel_hots, range(len(self.train_rel_hots)))

        for i, (verb_vec, noun_vec, rel_hot, idx) in enumerate(train_data):
            # print(verb_vec, noun_vec, rel_hot, idx)
            if i % 1000 == 0 and i > 0:
                print(i)
            creator.add_row(
                'train', verb_vec=verb_vec, noun_vec=noun_vec, rel=rel_hot, idx=idx)

        # Add the test data to H5PY
        test_data = zip(self.test_verb_vecs, self.test_noun_vecs,
                         self.test_rel_hots, range(len(self.test_rel_hots)))

        for verb_vec, noun_vec, rel_hot, idx in test_data:
            # print(verb_vec, noun_vec, rel_hot, idx)
            creator.add_row(
                'test', verb_vec=verb_vec, noun_vec=noun_vec, rel=rel_hot, idx=idx)

        creator.close()
