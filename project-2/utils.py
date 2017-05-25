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



# import argparse

# import numpy as np

# from datasets import AstroData
# from utils.dataset import H5PYDatasetCreator
# import utils


# def main(source, target):

#     data = AstroData(source)
#     creator = H5PYDatasetCreator(target)

#     any_data = data.get_item()

#     clean_shape = any_data.clean.shape

#     N = 64
#     num_examples = len(data)
#     creator.add_split('train', num_examples)
#     creator.add_source('clean', clean_shape, np.float32)
#     creator.add_source('dirtyimage', clean_shape, np.float32)
#     creator.add_source('dirtybeam', clean_shape, np.float32)

#     mat = None
#     invmat = None
#     cur_array = None

#     for i, datum in enumerate(data):
#         print(i)
#         # clean = datum.clean
#         # clean  = (clean / 255).astype(np.float32)
#         # vis, _ = datum.grid(fov_scale=4)
#         # _, weight = datum.grid()
#         # dirty = utils.uvtoimage(vis, crop=clean_shape).real.astype(np.float32)
#         # creator.add_row('train', clean=clean, dirty=dirty, weight=weight)

#         if cur_array != datum.array_id:
#             u, v = datum.get('u', 'v')
#             mat = utils.dftmat(u, v, datum.fov, N)
#             u, v = np.hstack([u, -u]), np.hstack([v, -v])
#             invmat = utils.dftmat(u, v, datum.fov, N, d=1)
#             cur_array = datum.array_id

#         clean = datum.clean / 255.0
#         vis = mat @ clean.ravel()
#         vis = np.hstack([vis, vis.conjugate()])
#         dirty_image = (invmat @ vis).reshape(N, N).real
#         dirty_beam = (invmat @ np.ones(vis.shape)).reshape(N, N).real

#         creator.add_row(
#             'train', clean=clean, dirtyimage=dirty_image, dirtybeam=dirty_beam)
