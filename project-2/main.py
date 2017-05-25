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

logging.basicConfig(level=logging.INFO)


class Config():
    # Picked data model file locations
    pairs = "storage/pairs.p"
    id2word = "storage/id2word.p"
    id2vector = "storage/id2vector.p"
    id2rel = "storage/id2rel.p"

    # The shape of each vector in the word2vec model
    vec_shape = (300,)

    # Keep the N most frequent relationships and discard the rest
    rel_cutoff = 29


def main():
    config = Config()

    dataModel = DataModel(config)
    
    dataModel.split_test_train()
    print('van de een anar de ander')
    dataModel.store_H5PY()
    # Note: only keep the top 29 relationships
    # counts = Counter([r for v, n, r in dataModel.pairs])
    # s_list = sorted(counts.items(), key=lambda x: -x[1])
    # keep = s_list[:29]

    # print(dataModel.get_batch(size=10000))



if __name__ == '__main__':
    main()
