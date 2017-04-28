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
    def __init__(self, K, tol=1e-6, max_iter=10_000, verbose=True):
        self.K = K
        self._loglikelihood = [float('-inf')]
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

    def _initialize_distribution(self, shape, norm_axis=0):
        dist = (np.random.rand(*shape) + 1) / 2
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

        for i in range(self.max_iter):

            self.step()

            ll_diff = self._loglikelihood[-1] - self._loglikelihood[-2]

            if self.verbose:
                print("[{}] LL diff: {} LL: {}".format(
                    i, ll_diff, self._loglikelihood[-1]))

            # Assert that loglikelihood does not decrease (We allow for a very
            # small decrease which my happen due to numerical instabilities
            # just before convergence).
            assert ll_diff >= -1e-11, "Loglikelihood should never decrease"

            # If update in Loglikelihood is smaller than tolerance, stop
            # procedure.
            if ll_diff < self.tol:
                break

    def step(self):
        self.estep()
        self.mstep()

    def estep(self):
        self.pc_vn = self.update_pc_vn(self._pairs)

    def update_pc_vn(self, pairs, update_loglikelihood=True):
        v_idx, n_idx = pairs.T
        pc_vn_ = self.sigma[:, None] * self.phi[:, v_idx] * self.lamb[:, n_idx]

        # At this point we can cheaply compute the log likelihood of the data
        # for the previous iteration (i.e., the LL obtained after the last
        # M-step). Hence, we compute the LL here to avoid duplicate
        # computation.
        if update_loglikelihood:
            self._loglikelihood.append(np.sum(
                self._pair_frequencies * np.log(pc_vn_.sum(axis=0))))

        return pc_vn_ / pc_vn_.sum(axis=0)

    def mstep(self):
        self.sigma = self.update_sigma()
        self.phi = self.update_phi()
        self.lamb = self.update_lambda()

    def update_sigma(self):
        sigma_ = (self._pair_frequencies * self.pc_vn).sum(axis=1)
        sigma_ = sigma_ / self.M
        return sigma_

    def update_phi(self):
        phi_ = np.empty((self.K, self.V))
        for i in range(self.K):
            weights = self._pair_frequencies * self.pc_vn[i]
            phi_[i] = np.bincount(self._pairs[:, 0], weights=weights)

        phi_ = phi_ / (self.M * self.sigma[:, None])
        return phi_

    def update_lambda(self):
        lamb_ = np.empty((self.K, self.N))
        for i in range(self.K):
            weights = self._pair_frequencies * self.pc_vn[i]
            lamb_[i] = np.bincount(self._pairs[:, 1], weights=weights)

        lamb_ = lamb_ / (self.M * self.sigma[:, None])
        return lamb_
