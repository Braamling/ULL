import numpy as np
import warnings


class VerbClassModel:

    def __init__(self, vnpair):
        self.vnpair = vnpair
        self.ll = [float('-inf')]

        self.sigma_diff = 0
        self.phi_diff = 0
        self.lamb_diff = 0

    def step(self):
        self.estep()
        self.mstep()

        total_update = self.sigma_diff + self.phi_diff + self.lamb_diff
        ll_diff = self.ll[-1] - self.ll[-2]
        print("ll diff: {} Total update: {}".format(ll_diff, total_update))

    def estep(self):
        self.update_pc_vn()

    def update_pc_vn(self):
        n_pairs = self.vnpair.n_pairs
        pc_vn_new = np.tile(self.vnpair.sigma[:, None], (1, n_pairs))
        pc_vn_new *= self.vnpair.phi[:, self.vnpair.pair_verbs]
        pc_vn_new *= self.vnpair.lamb[:, self.vnpair.pair_nouns]

        self.ll.append(np.sum(np.log(pc_vn_new.sum(axis=0))))

        pc_vn_new /= pc_vn_new.sum(axis=0)
        self.vnpair.pc_vn = pc_vn_new

    def mstep(self):
        self.update_sigma()
        warnings.warn('Update phi and lambda are disabled')
        # self.update_phi()
        # self.update_lambda()

    def update_sigma(self):
        sigma_new = np.sum(self.vnpair.pc_vn * self.vnpair.pair_freq, axis=1)
        sigma_new = sigma_new / self.vnpair.M

        print(sigma_new.sum())
        self.sigma_diff = np.abs(self.vnpair.sigma - sigma_new).sum()
        self.vnpair.sigma = sigma_new

    def update_phi(self):
        pc_vn = self.vnpair.pc_vn
        pair_freq = self.vnpair.pair_freq
        new_phi = np.zeros((self.vnpair.config.K, self.vnpair.n_verbs))
        for i, v in enumerate(self.vnpair.pair_verbs):
            new_phi[:, v] += pair_freq[i] * pc_vn[:, i]

        new_phi = new_phi / (self.vnpair.M * self.vnpair.sigma[:, None])
        self.phi_diff = np.abs(self.vnpair.phi - new_phi).sum()
        self.vnpair.phi = new_phi

    def update_lambda(self):
        pc_vn = self.vnpair.pc_vn
        pair_freq = self.vnpair.pair_freq
        new_lamb = np.zeros((self.vnpair.config.K, self.vnpair.n_nouns))
        for i, n in enumerate(self.vnpair.pair_nouns):
            new_lamb[:, n] += pair_freq[i] * pc_vn[:, i]

        new_lamb = new_lamb / (self.vnpair.M * self.vnpair.sigma[:, None])
        self.lamb_diff = np.abs(self.vnpair.lamb - new_lamb).sum()
        self.vnpair.lamb = new_lamb
