"""
# Unsupervised Language Learning (ULL) 2017 - project 2

### Main
Main file for configuration and examples

### Authors
Jorn Peters & Bram van den Akker

**Compatiable for both python2.7 and 3.0 but the pickle files are not
  transferable**
"""
import logging

from DataModel import DataModel
from collections import Counter
import operator


logging.basicConfig(level=logging.INFO)


class Config():
    # Picked data model file locations
    pairs = "storage/pairs.p"
    id2word = "storage/id2word.p"
    id2vector = "storage/id2vector.p"
    id2rel = "storage/id2rel.p"


def main():
    config = Config()

    dataModel = DataModel(config)


    # Note: only keep the top 29 relationships
    # counts = Counter([r for v, n, r in dataModel.pairs])
    # s_list = sorted(counts.items(), key=lambda x: -x[1])
    # keep = s_list[:29]

    print(dataModel.get_batch(size=10000))



if __name__ == '__main__':
    main()
