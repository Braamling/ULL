"""
# Unsupervised Language Learning (ULL) 2017 - project 1

### Main
Main file for configuration and examples

### Authors
Jorn Peters & Bram van den Akker

**Compatiable for both python2.7 and 3.0 but the pickle files are not
  transferable**
"""
from VNPairs import VNPairs


class Config():
    # location to store all pickle files
    storage = 'storage/vncounts'

    # File containing all training data. 
    # file should be of format: <verb> <noun> \n
    data = 'data/all_pairs'

    # Flag for refreshing the cache. 
    cache_refresh = True

    # Number of classes to be created
    K = 12


def main():
    config = Config()

    # Init VNPairs containing all processed data from the data source
    vnPairs = VNPairs(config)

    # Init the EM parameters
    sigma, phi, lamb = vnPairs.init_parameters()


if __name__ == '__main__':
    main()
