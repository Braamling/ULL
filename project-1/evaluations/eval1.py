import sys
import argparse
import logging
import datetime
from itertools import product, chain
from types import SimpleNamespace

import numpy as np

from VNPairs import VNPairs
from models.verbclass import VerbClassModel

LOG_FILENAME = "results/{}_results_eval1.log".format(
    datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
logging.basicConfig(level=logging.INFO, filename=LOG_FILENAME)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def main(datafile, nclusters, nruns, max_iter):

    config = SimpleNamespace(storage='storage/vncounts',
                             data=datafile,
                             cache_refresh=True)
    vnPairs = VNPairs(config)
    test_samples = vnPairs.extract_testset()

    n_test_samples = len(test_samples)
    test_samples = list(chain(*[[(v, n), (v_, n)]
                                for v, n, v_ in test_samples]))

    for clusters, iters in product(nclusters, max_iter):
        for run in range(nruns):
            logging.info('Run: clusters={}, iters={}, run={}'.format(
                clusters, iters, run))
            model = VerbClassModel(clusters, max_iter=iters, verbose=False)
            model.fit(vnPairs.pairs)
            # evaluate model
            p_n_v = model.p_noun_given_verb(test_samples).reshape(-1, 2)
            count_correct = np.sum(p_n_v[:, 0] >= p_n_v[:, 1])
            accuracy = count_correct / n_test_samples
            logging.info('Result: accuracy={}'.format(accuracy))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str)
    parser.add_argument('--nclusters', nargs='+', type=int, default=[30])
    parser.add_argument('--nruns', type=int, default=3)
    parser.add_argument('--max-iter', nargs='+', type=int, default=[50])

    args = parser.parse_args()
    main(**vars(args))
