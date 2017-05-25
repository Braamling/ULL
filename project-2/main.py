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


def main():
    config = Config()

    dataModel = DataModel(config)
    print("loaded")
    print(dataModel.pairs[0])
    print(dataModel.get_batch(size=1000))



if __name__ == '__main__':
    main()
