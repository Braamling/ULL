import numpy as np
import h5py

from fuel.datasets.hdf5 import H5PYDataset


class DatasetCreationException(Exception):
    pass


class H5PYDatasetCreator:

    def __init__(self, path):
        self._path = path
        self._sources = {}
        self._splits = {}
        self._setup_done = False

    def add_source(self, name, shape, dtype):
        if self._setup_done:
            msg = "It is not possible to add sources after adding data"
            raise DatasetCreationException(msg)
        self._sources[name] = dict(name=name, shape=shape, dtype=dtype)

    def add_split(self, name, num_examples):
        if self._setup_done:
            msg = "It is not possible to add splits after adding data"
            raise DatasetCreationException(msg)
        self._splits[name] = dict(name=name,
                                  num_examples=num_examples,
                                  offset=None,
                                  count=0)

    def _create_dataset(self):
        if self._setup_done:
            return

        num_examples = 0
        split_dict = {}
        for split in self._splits.values():
            split['offset'] = num_examples
            num_examples += split['num_examples']

            start = split['offset']
            end = start + split['num_examples']
            name = split['name']
            split_dict[name] = {s: (start, end) for s in self._sources}

        self._file = h5py.File(self._path, mode='w')
        for source in self._sources.values():
            name = source['name']
            shape = (num_examples, ) + source['shape']
            dtype = source['dtype']
            self._file.create_dataset(name, shape, dtype)

        self._file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        self._setup_done = True

    def add_row(self, split, **sources):
        if not self._setup_done:
            self._create_dataset()

        split_dict = self._splits[split]
        if split_dict['count'] >= split_dict['num_examples']:
            msg = "Split '{}' is already full".foramt(split)
            raise DatasetCreationException(msg)

        index = split_dict['offset'] + split_dict['count']
        for source, value in sources.items():
            self._file[source][index] = value

        split_dict['count'] += 1

    def close(self):
        if self._setup_done:
            self._file.flush()
            self._file.close()
