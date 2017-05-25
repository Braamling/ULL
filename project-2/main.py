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

# from DataModel import DataModel
from EvaluateModel import EvaluateModel

logging.basicConfig(level=logging.INFO)


class Config():
    # Picked data model file locations
    pairs = "storage/pairs.p"
    id2word = "storage/id2word.p"
    id2vector = "storage/id2vector.p"
    id2rel = "storage/id2rel.p"
    train_pairs = "storage/train_pairs"
    test_pairs = "storage/test_pairs"

    # The shape of each vector in the word2vec model
    vec_shape = (300,)

    # Keep the N most frequent relationships and discard the rest
    rel_cutoff = 29


def main():
    config = Config()

    dataModel = DataModel(config)
    
    dataModel.split_test_train()

    dataModel.store_H5PY()
    
    # evaluateModel = EvaluateModel(config)

    # verbs, nouns, rels = evaluateModel.convert_pairs_to_vectors(evaluateModel.test_pairs)
    # print(verbs.shape)
    # print(nouns.shape)
    # print(rels.shape)


if __name__ == '__main__':
    main()
