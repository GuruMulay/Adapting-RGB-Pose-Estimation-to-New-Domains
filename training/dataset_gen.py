import numpy as np
import os
import random
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize

from py_eggnog_server.py_eggnog_transformer import Transformer, AugmentSelection
from py_eggnog_server.py_eggnog_heatmapper import Heatmapper

import sys
sys.path.append("..")
from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig

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
        
        print("data_gen object paf_n_channels, hm_n_channels", paf_n_channels, hm_n_channels)
    
    
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
        
        
    def generate_with_masks(self, file_IDs, n_stages, shuffle=True, augment=True, mode="", online_aug=False, masking=False, map_to_coco=False, imagenet_dir="", crop_to_square=False):
        'Generates batches of samples'
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(file_IDs, shuffle)
            
            # Generate batches
            imax = int(len(indexes)/self.batch_size)  # len(indexes) == len(file_IDs)  # e.g., imax = 608/32 = 19
            for i in range(imax):
                # Find list of IDs
                file_IDs_temp = [file_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                
                # Generate data
                if online_aug:
                    X, y1, y2, kp = self.__data_generation_online(file_IDs_temp, augment, mode, map_to_coco)
                else:
                    # version v0 where hm had shape of (30, 40, 20) and paf had a shape of (30, 40, 36). This required generation of backgound hm on the fly for common_joints case, as that common background hm was not available in the e.npy file
                    # X, y1, y2, kp = self.__data_generation_offline(file_IDs_temp, augment, mode, map_to_coco, imagenet_dir)
                    
                    # version v1 (30th July, 2018). This version loads the saved common bacground hm from the .npy files
                    X, y1, y2, kp = self.__data_generation_offline_v1(file_IDs_temp, augment, mode, map_to_coco, imagenet_dir)
                    
                """
                returns [x] = (batch_size, height (240), width (320), 3)
                        [y1] = (batch_size, height (30), width (40), 36) pafs
                        [y2] = (batch_size, height (30), width (40), 20) heatmaps
                """
                
                # following is inefficient for one-branched networks because we read both y1 and y2 even if we want to use only one of them
                
                if crop_to_square:
                    X = X[:, :, 40:320-40, :]
                    y1 = y1[:, :, 5:35, :]
                    y2 = y2[:, :, 5:35, :]
                
                # save the batch data itself
                if self.save_transformed_path:
                    idx = np.random.randint(1000)
                    np.save(self.save_transformed_path + "/" + str(idx) + '_240x320_transformed.npy', X)  # (5, 240, 320, 3)
                    np.save(self.save_transformed_path + "/" + str(idx) + '_paf30x40_transformed.npy', y1)  # (5, 30, 40, 11)
                    np.save(self.save_transformed_path + "/" + str(idx) + '_heatmap30x40_transformed.npy', y2)  # (5, 30, 40, 11)
#                 print("dtypes:")
#                 print("X", X.shape, type(X), X.dtype)
#                 print("y1", y1.shape, type(y1), y1.dtype)
#                 print("y2", y2.shape, type(y2), y2.dtype)
#                 X (5, 240, 320, 3) <class 'numpy.ndarray'> uint8
#                 y1 (5, 30, 40, 12) <class 'numpy.ndarray'> float64
#                 y2 (5, 30, 40, 11) <class 'numpy.ndarray'> float64

                
                if not masking:
                    if self.branch_flag == 2:
                        yield [X], [y2] * n_stages

                    elif self.branch_flag == 1:
                        yield [X], [y1] * n_stages

                    else:
                        yield [X], [y1, y2] * n_stages
                
                else:  # to match with COCO's generator, use np.ones as a mask for both paf and 
                    if self.branch_flag == 2:
                        yield [X, np.ones(y2.shape)], [y2] * n_stages

                    elif self.branch_flag == 1:
                        yield [X, np.ones(y1.shape)], [y1] * n_stages

                    else:
                        yield [X, np.ones(y1.shape), np.ones(y2.shape)], [y1, y2] * n_stages
                    
                    
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

    
    def __data_generation_online(self, file_IDs_temp, augment, mode, map_to_coco):
        raise  NotImplementedError("This method needs to be updated for the joint removal option in the main file. Also map_to_coco is not implemented. Imagenet data reading not implemented.")
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
            
            # v2  [IMP: converting RGB to BGR to match with coco's BGR order]
            kpi = np.load(os.path.join(self.data_path, ID + '.npy'))
            X[i, :, :, :], y1[i, :, :, :], y2[i, :, :, :], kp[i, :] = self.transform_data_v1(
                                                    skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))[:,:,::-1],
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
    
    
    def __data_generation_offline(self, file_IDs_temp, augment, mode, map_to_coco, imagenet_dir):
        """
        Offline version where data is read from the *_augmented folders
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
            
            # if imagenet file
            if "train_set_" in ID:
                imagenet_img = skimage.io.imread(os.path.join(imagenet_dir, ID))
                if len(imagenet_img.shape) != 3:
                    X[i, :, :, :] = np.zeros((self.height, self.width, self.n_channels))
                    
                elif imagenet_img.shape[0] > self.height and imagenet_img.shape[1] > self.width:  # can crop if shapes of read image are more than req shape
                    # print("image shape", imagenet_img.shape)
                    temp_img = imagenet_img[0:self.height, 0:self.width, :]
                    # print("temp_img shape", temp_img.shape)
                    X[i, :, :, :] = temp_img[:,:,::-1]  # [IMP: converting RGB to BGR to match with coco's BGR order]
                    
                else:
                    temp_img = resize(imagenet_img, (self.height, self.width))
                    # print("temp_img shape", temp_img.shape)
                    X[i, :, :, :] = temp_img[:,:,::-1]  # [IMP: converting RGB to BGR to match with coco's BGR order]
                    
                hm_no_bk = np.zeros((self.hm_height, self.hm_width, self.hm_n_channels-1))
                y2[i, :, :, :] = np.dstack(( hm_no_bk, (1 - np.max(hm_no_bk[:,:,:], axis=2)) ))
                
                y1[i, :, :, :] = np.zeros((self.paf_height, self.paf_width, self.paf_n_channels))
            
            else:
                # load stored, augmented images and ground truth
                X[i, :, :, :] = skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))[:,:,::-1]  # [IMP: converting RGB to BGR to match with coco's BGR order]

                # Stored ground truths
                paf_temp = np.load(os.path.join(self.data_path, ID + '_paf30x40.npy'), mmap_mode='r')
                hm_temp = np.load(os.path.join(self.data_path, ID + '_heatmap30x40.npy'), mmap_mode='r')

                ## BUG FIX: Final -1 background heatmap needs to be updated for these set of joints
                hm_no_bk = hm_temp[:, :, np.array(EggnogGlobalConfig.joint_indices[:-1])]  # slice the loaded array using updated joint indices minus the background hm and # generate background heatmap
                if map_to_coco:
                    hm_no_bk_coco = hm_no_bk[:, :, EggnogGlobalConfig.eggnog_to_coco_10_joints_mapping]
                    y2[i, :, :, :] = np.dstack(( hm_no_bk_coco, (1 - np.max(hm_no_bk_coco[:,:,:], axis=2)) ))

                else:
                    y2[i, :, :, :] = np.dstack(( hm_no_bk, (1 - np.max(hm_no_bk[:,:,:], axis=2)) ))

                y1[i, :, :, :] = paf_temp[:, :, np.array(EggnogGlobalConfig.paf_indices_xy)]  # slice the loaded array using updated paf indices
            
                # y1[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_paf30x40.npy'))
                # y2[i, :, :, :] = np.load(os.path.join(self.data_path, ID + '_heatmap30x40.npy'))
                
            kp = None  # no need to read kp because it's not used after this line

            # save transformed 
#             if self.save_transformed_path:
# #                 print("saving ", ID.split('/')[-1] + '_240x320_transformed.jpg')
#                 skimage.io.imsave(self.save_transformed_path + "/" + ID.split('/')[-1] + '_240x320_transformed.jpg', X[i, :, :, :])
#                 np.save(self.save_transformed_path + "/" + ID.split('/')[-1] + '_paf30x40_transformed.npy', y1[i, :, :, :])
#                 np.save(self.save_transformed_path + "/" + ID.split('/')[-1] + '_heatmap30x40_transformed.npy', y2[i, :, :, :])
                    
            
        return X, y1, y2, kp
    
    
    def __data_generation_offline_v1(self, file_IDs_temp, augment, mode, map_to_coco, imagenet_dir):
        """
        Offline version where data is read from the *_Aug folders
        """
        # zero init helps in verifying dimensions of input image? during assignment?
        X = np.empty((self.batch_size, self.height, self.width, self.n_channels), dtype=np.uint8)
        y1 = np.empty((self.batch_size, self.paf_height, self.paf_width, self.paf_n_channels))  # 18 channels
        y2 = np.empty((self.batch_size, self.hm_height, self.hm_width, self.hm_n_channels))  # 11 channels
        # kp = np.empty((self.batch_size, (self.hm_n_channels-1)*2))  # (batch x (11-1)*2)  # e.g., 5x20
        
        # X = rgb images of original resolution (batch_size,240x320x3); y1 = pafs (batch_size,30x40x46); y2 = heatmaps (batch_size,30x40x25)
        
        # Generate data
        for i, ID in enumerate(file_IDs_temp):
            # print(ID)  # s07_1video/part1_layout_p14/20151116_230338_00_Video/# 20151116_230338_00_Video_vfr_105_skfr_105
            # if imagenet file
            if "train_set_" in ID:
                imagenet_img = skimage.io.imread(os.path.join(imagenet_dir, ID))
                if len(imagenet_img.shape) != 3:
                    X[i, :, :, :] = np.zeros((self.height, self.width, self.n_channels))
                    
                elif imagenet_img.shape[0] > self.height and imagenet_img.shape[1] > self.width:  # can crop if shapes of read image are more than req shape
                    # print("image shape", imagenet_img.shape)
                    temp_img = imagenet_img[0:self.height, 0:self.width, :]
                    # print("temp_img shape", temp_img.shape)
                    X[i, :, :, :] = temp_img[:,:,::-1]  # [IMP: converting RGB to BGR to match with coco's BGR order]
                    
                else:
                    temp_img = resize(imagenet_img, (self.height, self.width))
                    # print("temp_img shape", temp_img.shape)
                    X[i, :, :, :] = temp_img[:,:,::-1]  # [IMP: converting RGB to BGR to match with coco's BGR order]
                    
                hm_no_bk = np.zeros((self.hm_height, self.hm_width, self.hm_n_channels-1))
                y2[i, :, :, :] = np.dstack(( hm_no_bk, (1 - np.max(hm_no_bk[:,:,:], axis=2)) ))
                
                y1[i, :, :, :] = np.zeros((self.paf_height, self.paf_width, self.paf_n_channels))
            
            else:
                # load stored, augmented images and ground truth
                X[i, :, :, :] = skimage.io.imread(os.path.join(self.data_path, ID + '_240x320.jpg'))[:,:,::-1]  # [IMP: converting RGB to BGR to match with coco's BGR order]

                # Stored ground truths
                paf_temp = np.load(os.path.join(self.data_path, ID + '_paf30x40.npy'), mmap_mode='r')  # 46 channels
                hm_temp = np.load(os.path.join(self.data_path, ID + '_heatmap30x40.npy'), mmap_mode='r')  # 25 channels
                
                paf_all = paf_temp[:, :, np.array(EggnogGlobalConfig.paf_indices_xy)]  # 18 channels
                hm_with_bk = hm_temp[:, :, np.array(EggnogGlobalConfig.joint_indices)]  # slice the loaded array using updated joint indices with the common coco joints background hm
                
                
                if map_to_coco:
                    y1[i, :, :, :] = paf_all[:, :, EggnogGlobalConfig.eggnog18_to_coco_18_pafs_mapping]  # 18 channels
                    y2[i, :, :, :] = hm_with_bk[:, :, EggnogGlobalConfig.eggnog10_to_coco_10_joints_mapping + [-1]]  # 11 channels
                else:
                    y1[i, :, :, :] = paf_all
                    y2[i, :, :, :] = hm_with_bk

                # y1[i, :, :, :] = paf_temp[:, :, np.array(EggnogGlobalConfig.paf_indices_xy)]  # slice the loaded array using updated paf indices
            
            kp = None  # no need to read kp because it's not used after this line

        return X, y1, y2, kp
    
