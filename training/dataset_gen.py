import numpy as np
import os
import random
import matplotlib.pyplot as plt
import skimage.io

from py_eggnog_server.py_eggnog_transformer import Transformer, AugmentSelection

"""
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, data_path, height = 240, width = 320, n_channels = 3, batch_size = 5,
                 paf_height = 30, paf_width = 40, paf_n_channels = 36,
                 hm_height = 30, hm_width = 40, hm_n_channels = 20):
        
        'Initialization'
        self.data_path = data_path
        
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.batch_size = batch_size
        # self.shuffle = shuffle
        
        self.paf_height = paf_height
        self.paf_width = paf_width
        self.paf_n_channels = paf_n_channels
        
        self.hm_height = hm_height
        self.hm_width = hm_width
        self.hm_n_channels = hm_n_channels
        
        
    def generate(self, file_IDs, n_stages, shuffle=True, augment=True):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(file_IDs, shuffle)
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size)  # len(indexes) == len(file_IDs)  # e.g., imax = 608/32 = 19
            # for this epoch go through 608 images with 19 steps (imax) of 32 batch_size
            for i in range(imax):
                # Find list of IDs
                file_IDs_temp = [file_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                
                # Generate data
                X, y1, y2, kp = self.__data_generation(file_IDs_temp, augment)
                
                """
                returns [x] = (batch_size, height (240), width (320), 3)
                        [y1] = (batch_size, height (30), width (40), 36) pafs
                        [y2] = (batch_size, height (30), width (40), 20) heatmaps
                """
#                 yield [X], [y1, y2,
#                             y1, y2,
#                             y1, y2,
#                             y1, y2,
#                             y1, y2,
#                             y1, y2]
                
                yield [X], [y1, y2] * n_stages
        
        
    def transform_data(self, img, label_paf, label_hm, kp, augment):

        aug = AugmentSelection.random() if augment else AugmentSelection.unrandom()
#         print("transform data: before transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)  
        # transform data: before transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
        # transform data: after transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
        img, label_paf, label_hm, kp = Transformer.transform(img, label_paf, label_hm, kp, aug=aug)
#         print("transform data: after transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)

        return img, label_paf, label_hm, kp

    
    def __get_exploration_order(self, file_IDs, shuffle):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(file_IDs))
        if shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    
    def __data_generation(self, file_IDs_temp, augment):
        'Generates data of batch_size samples' 
        # X: (n_samples == batch_size, height (240), width (320), n_channels (3) (rgb or bgr))
        
        # Initialization
        # zero init helps in verifying dimensions of input image? during assignment?
        X = np.empty((self.batch_size, self.height, self.width, self.n_channels))
        y1 = np.empty((self.batch_size, self.paf_height, self.paf_width, self.paf_n_channels))
        y2 = np.empty((self.batch_size, self.hm_height, self.hm_width, self.hm_n_channels))
        kp = np.empty((self.batch_size, (self.hm_n_channels-1)*2))  # (batch x (20-1)*2)  # e.g., 5x38
        
        # X = rgb images of original resolution (batch_size,240x320x3); y1 = pafs (batch_size,30x40x36); y2 = heatmaps (batch_size,30x40x20)
        
        # Generate data
        for i, ID in enumerate(file_IDs_temp):
#             # toy dataset
#             # Store volume
#             X[i, :, :, :] = skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))
#             # print("img shape (h, w, c)", img.shape)  # height, width, channels (rgb)
#             # Store ground truths
#             y1[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_paf30_40.npy'))
#             y2[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_heatmap30_40.npy'))
            
            # original dataset
#             # v1
#             X[i, :, :, :] = skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))
#             # print("img shape (h, w, c)", img.shape)  # height, width, channels (rgb)
#             # Store ground truths
#             y1[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_paf30x40.npy'))
#             y2[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_heatmap30x40.npy'))
            
#             # option 2
#             # verify dimensions of input image with parameters
#             # append images to X
#             # append labels to y1 and y2
            
            
#             # load the keypoints as well
#             kp[i, :] = np.load(os.path.join(self.data_path, ID + '.npy'))
#             print("kp shape", kp.shape)  # kp shape (38,) 19 joints and x, y
            
#             # transform the laoded images and corresponding labels with the same transformation
#             X, y1, y2, kp = self.transform_data(X, y1, y2, kp, augment)
            
            # v2
            X[i, :, :, :], y1[i, :, :, :], y2[i, :, :, :], kp[i, :] = self.transform_data(
                                                    skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg')),
                                                    np.load(os.path.join(self.data_path, ID + '_paf30x40.npy')),
                                                    np.load(os.path.join(self.data_path, ID + '_heatmap30x40.npy')),
                                                    np.load(os.path.join(self.data_path, ID + '.npy')),
                                                    augment
                                                    )  # loads individual images and npys, returns their transformed versions without changing shapes
            
            # print("sum of x[4, :, :, :]", i, np.sum(X[4, :, :, :]), np.sum(y1[4, :, :, :]), np.sum(y1[4, :, :, :]), np.sum(kp[4, :]))  # untimeWarning: overflow encountered in reduce  # also somehow kp is not summing to zero!
#             transform data: before transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
#             in class transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
#             returning transformed data and labels...
#             transform data: after transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)

            
        return X, y1, y2, kp
