"""
# Unsupervised Language Learning (ULL) 2017 - project 1

### Main
Main file for configuration and examples

### Authors
Jorn Peters & Bram van den Akker

**Compatiable for both python2.7 and 3.0 but the pickle files are not
  transferable**
"""
import logging

from VNPairs import VNPairs
from models.verbclass import VerbClassModel


logging.basicConfig(level=logging.INFO)


class Config():
    # location to store all pickle files
    storage = 'storage/vncounts'

    # File containing all training data.
    # file should be of format: <verb> <noun> \n
    data = 'data/all_pairs'
    # data = 'data/gold_deps.txt'

    # Flag for refreshing the cache.
    cache_refresh = True

    # Number of classes to be created
    K = 30


def main():
    config = Config()

    # Init VNPairs containing all processed data from the data source
    vnPairs = VNPairs(config)

    samples = vnPairs.extract_testset(output_size=100)

    print(samples)

    # Init the EM parameters
    # vnPairs.init_parameters()

    # model = VerbClassModel(config.K)
    # model.fit(vnPairs.pairs)

if __name__ == '__main__':
    main()
