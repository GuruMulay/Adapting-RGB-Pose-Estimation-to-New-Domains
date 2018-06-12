
import h5py
import random
import json
import numpy as np

from py_rmpe_server.py_rmpe_config import RmpeGlobalConfig, RmpeCocoConfig
from py_rmpe_server.py_rmpe_transformer import Transformer, AugmentSelection
from py_rmpe_server.py_rmpe_heatmapper import Heatmapper

class RawDataIterator:

    def __init__(self, h5file, shuffle = True, augment = True):

        self.h5file = h5file
        self.h5 = h5py.File(self.h5file, "r")
        self.datum = self.h5['datum']
        self.heatmapper = Heatmapper()
        self.augment = augment
        self.shuffle = shuffle

    def gen(self, dbg=False):

        keys = list(self.datum.keys())

        if self.shuffle:
            random.shuffle(keys)
        
        # print("keys len, keys[0:2]", len(keys), keys[0:2])  # keys len, keys[0:2] 117873 ['0111017', '0031207']

        for key in keys:

            image, mask, meta = self.read_data(key)
            debug = {}

            debug['img_path']=meta['img_path']
            debug['mask_miss_path'] = meta['mask_miss_path']
            debug['mask_all_path'] = meta['mask_all_path']
            
            # image (368, 368, 3)
            # print("mask_img shape", mask.shape)  # (424, 640)
            image, mask, meta, labels = self.transform_data(image, mask, meta)  # transforms and generates ground truth hms
            image = np.transpose(image, (2, 0, 1))
            
#             print("gen of raw data iterator")
#             print("data_image shape", image.shape)  # (3, 368, 368)
#             print("mask_img shape", mask.shape)  # (46, 46)
#             print("label shape", labels.shape)  # (57, 46, 46)
#             print("kpts shape", meta['joints'].shape, "type(meta)", type(meta))  # (n, 18, 3)  n varies (depending on n_persons) 3 => x, y, ?
            
            yield image, mask, labels, meta['joints']

    def num_keys(self):
        return len(list(self.datum.keys()))

    def read_data(self, key):

        entry = self.datum[key]

        assert 'meta' in entry.attrs, "No 'meta' attribute in .h5 file. Did you generate .h5 with new code?"

        meta = json.loads(entry.attrs['meta'])
        meta['joints'] = RmpeCocoConfig.convert(np.array(meta['joints']))
        data = entry.value

        if data.shape[0] <= 6:
            # TODO: this is extra work, should write in store in correct format (not transposed)
            # can't do now because I want storage compatibility yet
            # we need image in classical not transposed format in this program for warp affine
            data = data.transpose([1,2,0])

        img = data[:,:,0:3]
        mask_miss = data[:,:,4]
        mask = data[:,:,5]

        return img, mask_miss, meta

    def transform_data(self, img, mask, meta):

#         aug = AugmentSelection.random() if self.augment else AugmentSelection.unrandom()
        aug = AugmentSelection.unrandom()
        # print("transform data: before transform", img.shape)  
        # transform data: before transform (427, 640, 3)  # could be any shape
        # transform data: after transform (368, 368, 3)
        img, mask, meta = Transformer.transform(img, mask, meta, aug=aug)
        # print("transform data: after transform img shape ===========", img.shape)
        labels = self.heatmapper.create_heatmaps(meta['joints'], mask)

        return img, mask, meta, labels


    def __del__(self):

        self.h5.close()
