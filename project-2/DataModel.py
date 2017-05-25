"""
Unsupervised Language Learning (ULL) 2017 - project 2

### DataModel
This model is a helper class for loading and using the preprocessed dataset

### Authors
Jorn Peters & Bram van den Akker

"""

import pickle
import numpy as np

class DataModel():
    def __init__(self, config):
        self.config = config

        self.pairs = pickle.load(open(self.config.pairs, 'rb'))
        self.id2word = pickle.load(open(self.config.id2word, 'rb'))
        self.id2vector = np.asarray(pickle.load(open(self.config.id2vector, 'rb')))
        self.id2rel = pickle.load(open(self.config.id2rel, 'rb'))

        self.batch_pointer = 0

    def get_batch(self, size=1000):
        batch = self.pairs[self.batch_pointer: self.batch_pointer + size]

        # batch = [np.append(np.append(self.id2vector[verb], self.id2vector[noun]), np.asarray([rel])) for verb, noun, rel in batch]


        batch = [(self.id2vector[verb], self.id2vector[noun], rel)) for verb, noun, rel in batch]

        self.batch_pointer += size

        return batch

