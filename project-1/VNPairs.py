"""
Unsupervised Language Learning (ULL) 2017 - project 1

### VNPairs
Verb noun pair helper class. This class is used for preprocessing the
verb noun pair data files and store/load the preprocessed data to disk

### Authors
Jorn Peters & Bram van den Akker

**Compatiable for both python2.7 and 3.0 but the pickle files are not
  transferable**
"""

import numpy as np
import pickle


class VNPairs():
    def __init__(self, config):
        self.config = config

        # Load the data
        if not self.config.cache_refresh:
            self.load_cache()
        else:
            self.preprocess_data()
            self.store_cache()

    def load_cache(self):
        storage = pickle.load(open(self.config.storage, "rb"))

        # extract data
        self.n_verbs, self.n_nouns, self.pairs = storage["counts"]
        self.verb2id, self.id2verb, self.noun2id,\
            self.id2noun = storage["mappings"]

    def store_cache(self):
        # Structure data
        # TODO: save n_pairs, pair_verbs, pair_nouns, pair_freq
        storage = {
            "counts": [self.n_verbs, self.n_nouns, self.pairs],
            "mappings":
            [self.verb2id, self.id2verb, self.noun2id, self.id2noun]
        }

        pickle.dump(storage, open(self.config.storage, "wb"))

    def preprocess_data(self):
        self.pairs = {}

        # Load all data from the configured data file
        with open(self.config.data) as data:
            verbs = {}
            nouns = {}
            for pair in data:
                # Somehow each line has trailing whitespaces
                pair = pair.rstrip()

                # split verb and noun
                verb, noun = pair.split(" ", 1)

                # Set dict item for verb and noun count
                verbs[verb] = None
                nouns[noun] = None

                if pair in self.pairs:
                    self.pairs[pair] += 1
                else:
                    self.pairs[pair] = 1

            # Create the translation matrices
            self.id2verb = list(verbs.keys())
            self.id2noun = list(nouns.keys())

            # inverse mapping
            self.verb2id = {verb: idx for idx, verb in enumerate(self.id2verb)}
            self.noun2id = {noun: idx for idx, noun in enumerate(self.id2noun)}

            # Store the counts of verbs
            self.n_verbs = len(self.id2verb)
            self.n_nouns = len(self.id2noun)
            self.n_pairs = len(self.pairs)

            # Create list of indices for all verbs and nouns in ith pair.
            id2pairs = list(self.pairs.keys())
            splits = [k.split(" ", 1) for k in id2pairs]
            splits = [(self.verb2id[v], self.noun2id[n]) for v, n in splits]
            pair_verbs, pair_nouns = zip(*splits)

            self.pair_freq = np.array([self.pairs[k] for k in id2pairs])
            self.M = self.pair_freq.sum()
            print("M: ", self.M)
            self.pair_verbs = np.array(pair_verbs)
            self.pair_nouns = np.array(pair_nouns)

    def init_parameters(self):
        # Initialize the EM parameters
        self.sigma = np.random.rand(self.config.K)
        self.phi = np.random.rand(self.config.K, self.n_verbs)
        self.lamb = np.random.rand(self.config.K, self.n_nouns)
        self.pc_vn = None

        self.sigma = self.sigma / self.sigma.sum()
        self.phi = self.phi / self.phi.sum(axis=0)
        self.lamb = self.lamb / self.lamb.sum(axis=0)
