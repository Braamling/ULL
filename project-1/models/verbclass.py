import logging

import numpy as np

logger = logging.getLogger(__name__)


class WordIndex:
    def __init__(self, words):
        self._str2idx = {}
        self._idx2str = []

        for word in words:
            idx = self._str2idx.setdefault(word, len(self._str2idx))
            if idx == len(self._idx2str):
                self._idx2str.append(word)

    def __len__(self):
        return len(self._idx2str)

    def __getitem__(self, key):
        try:
            return self._idx2str[key]
        except TypeError:
            return self._str2idx[key]


class VerbClassModel:
    def __init__(self, K):
        self.K = K

    def _initialize_distribution(self, shape, norm_axis=0):
        dist = np.random.rand(*shape)
        dist = dist / np.expand_dims(dist.sum(axis=norm_axis), norm_axis)
        return dist

    def initialize_parameters(self):
        logger.info('Initialize VerbClassModel parameters')
        self.sigma = self._initialize_distribution((self.K, ))
        self.phi = self._initialize_distribution((self.K, self.V), 1)
        self.lamb = self._initialize_distribution((self.K, self.N), 1)
        self.pc_vn = None

    def _setup(self, pairs):
        pair_list = [k.split(' ', 1) + [v] for k, v in pairs.items()]
        verbs, nouns, frequencies = zip(*pair_list)

        self._noun_index = WordIndex(nouns)
        self._verb_index = WordIndex(verbs)
        self._pairs = np.array([(self._verb_index[v], self._noun_index[n])
                                for v, n, _ in pair_list])
        self._pair_frequencies = np.array(frequencies)

        self.M = self._pair_frequencies.sum()
        self.V = len(self._verb_index)
        self.N = len(self._noun_index)
        self.initialize_parameters()

    def fit(self, pairs):
        self._setup(pairs)

        while True:
            self.step()

    def step(self):
        self.estep()
        self.mstep()

        total_update = 456
        ll_diff = 123
        print("ll diff: {} Total update: {}".format(ll_diff, total_update))

    def estep(self):
        self.update_pc_vn()

    def update_pc_vn(self):
        pc_vn_new = self.sigma[:, None]
        pc_vn_new = pc_vn_new * self.phi[:, self._pairs[:, 0]]
        pc_vn_new = pc_vn_new * self.lamb[:, self._pairs[:, 1]]

        pc_vn_new = pc_vn_new / pc_vn_new.sum(axis=0)
        self.pc_vn = pc_vn_new

    def mstep(self):
        self.update_sigma()
        # self.update_phi()
        # self.update_lambda()

    def update_sigma(self):
        sigma_new = np.sum(self.pc_vn * self._pair_frequencies, axis=1)
        sigma_new = sigma_new / self.M

        self.sigma = sigma_new

    # def update_phi(self):
    #     pc_vn = self.vnpair.pc_vn
    #     pair_freq = self.vnpair.pair_freq
    #     new_phi = np.zeros((self.vnpair.config.K, self.vnpair.n_verbs))
    #     for i, v in enumerate(self.vnpair.pair_verbs):
    #         new_phi[:, v] += pair_freq[i] * pc_vn[:, i]

    #     new_phi = new_phi / (self.vnpair.M * self.vnpair.sigma[:, None])
    #     self.phi_diff = np.abs(self.vnpair.phi - new_phi).sum()
    #     self.vnpair.phi = new_phi

    # def update_lambda(self):
    #     pc_vn = self.vnpair.pc_vn
    #     pair_freq = self.vnpair.pair_freq
    #     new_lamb = np.zeros((self.vnpair.config.K, self.vnpair.n_nouns))
    #     for i, n in enumerate(self.vnpair.pair_nouns):
    #         new_lamb[:, n] += pair_freq[i] * pc_vn[:, i]

    #     new_lamb = new_lamb / (self.vnpair.M * self.vnpair.sigma[:, None])
    #     self.lamb_diff = np.abs(self.vnpair.lamb - new_lamb).sum()
    #     self.vnpair.lamb = new_lamb
