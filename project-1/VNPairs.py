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
import random
from collections import Counter


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
                if verb in verbs:
                    verbs[verb] += 1
                else:
                    verbs[verb] = 1

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

            # Create verbs sample array
            verbs =  {self.verb2id[k]:v for k,v in verbs.items()}
            self.samples_verbs = list(Counter(verbs).elements())

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

            self.pair_verbs = np.array(pair_verbs)
            self.pair_nouns = np.array(pair_nouns)

    def extract_testset(self, output_size=3000, min_freq=30, max_freq=3000):
        # Sample 3000 unique pairs
        test_pairs = []
        pair_verb_counts = Counter(self.pair_verbs)
        pair_noun_counts = Counter(self.pair_nouns)
        number2pair = list(self.pairs.keys())

        while len(test_pairs) < output_size:
            # Retrieve a random verb noun pair.
            pair, idx = self.sample_pair(number2pair)
            verb, noun = pair.split(" ", 1)
            i_verb, i_noun = self.verb2id[verb], self.noun2id[noun]

            # check the amount verb and noun > 1 in pair_verbs and pair_nouns 
            # else retry.
            noun_verb_count = [pair_verb_counts[i_verb],
                               pair_noun_counts[i_noun]]

            # Check whether both noun and verb still exist in dataset
            if all(i > 1 for i in noun_verb_count):
                i_verb_prime = self.sample_verb(i_verb, noun)

                # Check if all nouns and verbs are represented as required.
                if all(i > min_freq and i < max_freq for i in noun_verb_count):
                    test_pairs.append((self.id2verb[i_verb],
                                       self.id2noun[i_noun],
                                       self.id2verb[i_verb_prime]))

                    pair_verb_counts[i_verb] -= 1
                    pair_noun_counts[i_noun] -= 1
                    del self.pairs[pair]
                    del number2pair[idx]
                else:
                    test_pairs.append(None)

                if len(test_pairs) % 100 is 0:
                    print(len(test_pairs) )

        # Remove all Nones from list
        test_pairs = [i for i in test_pairs if i is not None]

        self.update_pairs()

        return test_pairs


    def sample_verb(self, verb, noun):
        n_verbs = list(self.samples_verbs)

        while True:
            i_verb = self.samples_verbs[random.randint(0, len(n_verbs) - 1)]
            pair = self.id2verb[i_verb] + " " + noun
            if i_verb is not verb and pair not in self.pairs:
                return i_verb

    def sample_pair(self, number2pair):
        idx = random.randint(0, len(number2pair) - 1)

        return number2pair[idx], idx

    def update_pairs(self):
        """ Recalculate all model parameters that depend on self.pairs
            after sampling.
        """ 
        self.n_pairs = len(self.pairs)
        id2pairs = list(self.pairs.keys())

        # Create list of indices for all verbs and nouns in ith pair.
        id2pairs = list(self.pairs.keys())
        splits = [k.split(" ", 1) for k in id2pairs]
        splits = [(self.verb2id[v], self.noun2id[n]) for v, n in splits]
        pair_verbs, pair_nouns = zip(*splits)

        self.pair_freq = np.array([self.pairs[k] for k in id2pairs])
        self.M = self.pair_freq.sum()

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
