import numpy as np
import os
import random
import matplotlib.pyplot as plt
import skimage.io

from py_eggnog_server.py_eggnog_transformer import Transformer, AugmentSelection
from py_eggnog_server.py_eggnog_heatmapper import Heatmapper

"""
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, data_path, height = 240, width = 320, n_channels = 3, batch_size = 5,
                 paf_height = 30, paf_width = 40, paf_n_channels = 36,
                 hm_height = 30, hm_width = 40, hm_n_channels = 20,
                 branch_flag = 0,
                 save_transformed_path = None):
        
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
        
        self.branch_flag = branch_flag
        
        self.save_transformed_path = save_transformed_path  # used to save the transformed images in this path if it is not null
#         print("===== self.save_transformed_path", self.save_transformed_path)
        if self.save_transformed_path:
            os.makedirs(self.save_transformed_path, exist_ok=True)
            
        # change this later to direct class calling
        self.heatmapper = Heatmapper()
    
    
    def generate_and_save(self, file_IDs, n_stages, shuffle=True, augment=True):
        'Generates batches of samples and save transformed version to a folder'
        print("in gen")
        indexes = self.__get_exploration_order(file_IDs, shuffle)
            
        # Generate batches
        imax = int(len(indexes)/self.batch_size)  # len(indexes) == len(file_IDs)  # e.g., imax = 608/32 = 19
        
        # for this epoch go through 608 images with 19 steps (imax) of 32 batch_size
        for i in range(imax):
            # Find list of IDs
            file_IDs_temp = [file_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                
            # Generate data
            X, y1, y2, kp = self.__data_generation(file_IDs_temp, augment)
        
    
    def generate(self, file_IDs, n_stages, shuffle=True, augment=True, mode="", online_aug=False):
        'Generates batches of samples'
        # print("in gen")
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
                if online_aug:
                    X, y1, y2, kp = self.__data_generation_online(file_IDs_temp, augment, mode)
                else:
                    X, y1, y2, kp = self.__data_generation_offline(file_IDs_temp, augment, mode)
                    
                """
                returns [x] = (batch_size, height (240), width (320), 3)
                        [y1] = (batch_size, height (30), width (40), 36) pafs
                        [y2] = (batch_size, height (30), width (40), 20) heatmaps
                """
                
                # following is inefficient for one-branched networks because we read both y1 and y2 even if we want to use only one of them
                if self.branch_flag == 2:
                    yield [X], [y2] * n_stages

                elif self.branch_flag == 1:
                    yield [X], [y1] * n_stages
                    
                else:
                    yield [X], [y1, y2] * n_stages
        
        
    def transform_data_v1(self, img, kp, augment):

        aug = AugmentSelection.random() if augment else AugmentSelection.unrandom()
#         print("transform data: before transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)  
        # transform data: before transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
        # transform data: after transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
        img, kp = Transformer.transform_v1(img, kp, aug=aug)
#         print("transform data: after transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)
#         print("aug =====", augment)

        label_paf, label_hm = self.heatmapper.get_pafs_and_hms_heatmaps_v1(kp)  # these kp are in the image space (240x320)
        
        return img, label_paf, label_hm, kp

    
    def transform_data(self, img, kp, kp_tracking_info, augment):
        """
        img: 240x320 RGB image
        kp: these are in 1920x1080 space (38, )  19*2 (x,y)
        kp_tracking_info: tracking info for 19 joints (19, )
        augment Boolean
        """

        aug = AugmentSelection.random() if augment else AugmentSelection.unrandom()
    #         print("transform data: before transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)  
            # transform data: before transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
            # transform data: after transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
        img, kp, kp_tracking_info = Transformer.transform(img, kp, kp_tracking_info, aug=aug)
    #         print("transform data: after transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)
        label_paf, label_hm = self.heatmapper.get_pafs_and_hms_heatmaps(kp, kp_tracking_info)  # these kp are in the image space (240x320)

        return img, label_paf, label_hm, kp
    
    
    def __get_exploration_order(self, file_IDs, shuffle):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(file_IDs))
        if shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    
    def __data_generation_online(self, file_IDs_temp, augment, mode):
        'Generates data of batch_size samples' 
        # X: (n_samples == batch_size, height (240), width (320), n_channels (3) (rgb or bgr))
        
        # Initialization
        # zero init helps in verifying dimensions of input image? during assignment?
        X = np.empty((self.batch_size, self.height, self.width, self.n_channels), dtype=np.uint8)
        y1 = np.empty((self.batch_size, self.paf_height, self.paf_width, self.paf_n_channels))
        y2 = np.empty((self.batch_size, self.hm_height, self.hm_width, self.hm_n_channels))
        kp = np.empty((self.batch_size, (self.hm_n_channels-1)*2))  # (batch x (20-1)*2)  # e.g., 5x38
        
        # X = rgb images of original resolution (batch_size,240x320x3); y1 = pafs (batch_size,30x40x36); y2 = heatmaps (batch_size,30x40x20)
        
        # Generate data
        for i, ID in enumerate(file_IDs_temp):
#             print(ID)  # s07_1video/part1_layout_p14/20151116_230338_00_Video/20151116_230338_00_Video_vfr_105_skfr_105

#             # toy dataset
#             # Store volume
#             X[i, :, :, :] = skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))
#             # print("img shape (h, w, c)", img.shape)  # height, width, channels (rgb)
#             # Store ground truths
#             y1[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_paf30_40.npy'))
#             y2[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_heatmap30_40.npy'))
            
#             # original dataset
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
            kpi = np.load(os.path.join(self.data_path, ID + '.npy'))
            X[i, :, :, :], y1[i, :, :, :], y2[i, :, :, :], kp[i, :] = self.transform_data_v1(
                                                    skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg')),
                                                    np.delete(kpi, np.arange(0, kpi.size, 3)), 
                                                    augment
                                                    )  # loads individual images and npys, returns their transformed versions without changing shapes
            
#             np.delete(sk_keypoints_with_tracking_info, np.arange(0, sk_keypoints_with_tracking_info.size, 3))  # this is without tracking info, by removing the tracking info
            
#             print("X[i, :, :, :], y1[i, :, :, :], y2[i, :, :, :],", X[i, :, :, :].shape, X[i, :, :, :].dtype, type(X[i, :, :, :]), y1[i, :, :, :].shape, y2[i, :, :, :].shape)
#             print("X", X[i, :, :, :])
            # save transformed # /s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/s07_1video_transformed_val/part1_layout_p14/20151116_230338_00_Video
            
            if self.save_transformed_path:
#                 print("saving ", ID.split('/')[-1] + '_240x320_transformed.jpg')
                skimage.io.imsave(self.save_transformed_path + "/" + ID.split('/')[-1] + '_240x320_transformed.jpg', X[i, :, :, :])
                np.save(self.save_transformed_path + "/" + ID.split('/')[-1] + '_paf30x40_transformed.npy', y1[i, :, :, :])
                np.save(self.save_transformed_path + "/" + ID.split('/')[-1] + '_heatmap30x40_transformed.npy', y2[i, :, :, :])
                    
            # print("sum of x[4, :, :, :]", i, np.sum(X[4, :, :, :]), np.sum(y1[4, :, :, :]), np.sum(y1[4, :, :, :]), np.sum(kp[4, :]))  # untimeWarning: overflow encountered in reduce  # also somehow kp is not summing to zero!
#             transform data: before transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
#             in class transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
#             returning transformed data and labels...
#             transform data: after transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)

            
        return X, y1, y2, kp
    
    
    def __data_generation_offline(self, file_IDs_temp, augment, mode):
        """
        Offline version where data is read from the *_augmneted folders
        """
        # X: (n_samples == batch_size, height (240), width (320), n_channels (3) (rgb or bgr))
        
        # Initialization
        # zero init helps in verifying dimensions of input image? during assignment?
        X = np.empty((self.batch_size, self.height, self.width, self.n_channels), dtype=np.uint8)
        y1 = np.empty((self.batch_size, self.paf_height, self.paf_width, self.paf_n_channels))
        y2 = np.empty((self.batch_size, self.hm_height, self.hm_width, self.hm_n_channels))
        kp = np.empty((self.batch_size, (self.hm_n_channels-1)*2))  # (batch x (20-1)*2)  # e.g., 5x38
        
        # X = rgb images of original resolution (batch_size,240x320x3); y1 = pafs (batch_size,30x40x36); y2 = heatmaps (batch_size,30x40x20)
        
        # Generate data
        for i, ID in enumerate(file_IDs_temp):
#             print(ID)  # s07_1video/part1_layout_p14/20151116_230338_00_Video/20151116_230338_00_Video_vfr_105_skfr_105

#             # option 2
#             # verify dimensions of input image with parameters
#             # append images to X
#             # append labels to y1 and y2
            
            # load stored, augmented images and ground truth
            X[i, :, :, :] = skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))
            # print("img shape (h, w, c)", img.shape)  # height, width, channels (rgb)
            # Stored ground truths
            y1[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_paf30x40.npy'))
            y2[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_heatmap30x40.npy'))
            kp = None  # no need to read kp because it's not used after this line

            # save transformed 
            if self.save_transformed_path:
#                 print("saving ", ID.split('/')[-1] + '_240x320_transformed.jpg')
                skimage.io.imsave(self.save_transformed_path + "/" + ID.split('/')[-1] + '_240x320_transformed.jpg', X[i, :, :, :])
                np.save(self.save_transformed_path + "/" + ID.split('/')[-1] + '_paf30x40_transformed.npy', y1[i, :, :, :])
                np.save(self.save_transformed_path + "/" + ID.split('/')[-1] + '_heatmap30x40_transformed.npy', y2[i, :, :, :])
                    
            
        return X, y1, y2, kp
