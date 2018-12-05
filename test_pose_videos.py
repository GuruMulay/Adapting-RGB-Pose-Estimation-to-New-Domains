import sys
import os
import numpy as np
sys.path.append("..")
from glob import glob
import pprint
from model import get_testing_model_eggnog_v1
# for testing 5000 eggnog images on rmpe
from model_rmpe_test import get_rmpe_test_model

# 
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import pylab as plt

import random
sys.path.append("./training/")
from imagenet_images import ImagenetImages
from sessions_processing import Session

from config_reader import config_reader
from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig
from py_eggnog_server.py_eggnog_transformer import keypoint_transform_to_240x320_image

# change
import util

from scipy.ndimage.filters import gaussian_filter

from operator import itemgetter


os.environ["CUDA_VISIBLE_DEVICES"]="0"

EXP_BASE_DIR = "/s/red/b/nobackup/data/eggnog_cpm/training_files/"
eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/"

imagenet_dir = '/s/red/a/nobackup/imagenet/images/train/'
eggnog_meta_dir = '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_meta/'

test_video = True

add_imagenet_images = True
imagenet_fraction = 0.0

eggnog_testing = True  # whether to use 320x240 or 368x368

calculate_loss = False  # keep this false for rmpe testing!!!!!!!
hm_save = False  # save predicted hms to disk
kp_save = True
img_save = True

verbose = False
verbose_pckh = False
verbose_pck = False

rmpe_testing = False
rmpe_weights_file = weights_path = "/s/parsons/h/proj/vision/usr/guru5/repos_cpm/keras_Realtime_Multi-Person_Pose_Estimation/model/keras/model.h5"
rmpe_to_eggnog_slicing = [0,1,2,3,4,5,6,7,8,11]

if rmpe_testing:
    np_branch1 = 38
    np_branch2 = 19
else:
    np_branch1 = 18  # 18 (keeping only common joints and paf pairs) # 38
    np_branch2 = 11  # 11 (keeping only common joints and paf pairs) # 19

paf_pairs_indices_10joints = [[0, 1], 
                    [2, 1], [3, 2], [4, 3],
                    [5, 1], [6, 5], [7, 6],
                    [8, 1], [9, 1]
                    ]

random_seed = 1

##### since kinect RGBskeleton.txt itself contains the locations of joints in the format of float pixel numbers instead of discrete pixel numbers, the ground truth is more precise with floats rather than ints 
### IMP whether to keep or not
# roundof_pred = True

import datetime
import time

def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%m%d%Y%H%M')


def eucl_loss_np(x, y):
    l = np.sum(np.square(x - y))  
    return l


class Test:
    
    def __init__(self, experiment_dir, epoch_num, video_path):
        
        self.experiment_dir = experiment_dir
        self.epoch_num = epoch_num
        
        # to test video only
        self.video_path = video_path  # /s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s18/part1_layouts_p36/20160129_192616_00_Video.avi
        self.video_name = self.video_path.split("/")[-1]  # 20160129_192616_00_Video.avi
        self.video_folder_name = self.video_name.split(".")[0]  # 20160129_192616_00_Video
        self.video_layout_name = self.video_path.split("/")[-2]  # part1_layouts_p36
        self.video_session_name = self.video_path.split("/")[-3]  # s18
        self.test_session_for_video = [str(self.video_path.split("/")[-3])]  # "s18" or "s19"
        
        self.BASE_DIR_TRAIN = os.path.join(EXP_BASE_DIR, self.experiment_dir + "training/weights_egg/")
        
        self.BASE_DIR_TEST_KP = ""
        self.BASE_DIR_TEST_HEATMAPS = ""
        self.BASE_DIR_TEST_RESULTS = ""
        self.BASE_DIR_TEST_IMAGES = ""
        
        self.BASE_DIR_TEST_VIDEO = ""
        
        # stores test data
        self.partition_dict = {}
        
        self.model = None
        
        # NETWORK PARAMAETERS
        self.n_stages = 2
        self.n_stages_rmpe = 2
        self.branch_flag = 0  # 0 => both branches; 1 => branch L1 only; 2 => branch L2 only (heatmaps only)
        
        # sessions
        ##### ========================================= #####
        ### train
        # without
        # ['s01', 's02', 's03', 's04', 's05', 's16', 's20']
        # with
        # ['s08', 's09', 's10', 's11', 's12', 's17', 's21']

        ### validation
        # without
        # ['s06', 's07']
        # with
        # ['s14', 's15']

        ### test
        # without
        # ['s18']
        # with
        # ['s19']
        ##### ========================================= #####
        self.test_sessions = ['s18', 's19']
        self.n_test_imgs = 10000
        self.len_test_set = 5000
        self.aug_fraction = 0.0  # use aug_fraction % of the images from aug set and remaining from original non_aug set
                
        # loss calc
        self.total_loss = 0
        self.total_test_samples = 0 
        
        # pck calculations
        self.metric_type = ["PCKh", "PCK", "AP"]
        self.n_hm = 10  # 10 joints
        
        # for pck
        self.b_box_method = "vertical_box" 
#         self.pck_at = [0.0, 0.02, 0.04, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]  # @_, @_, and @_
#         self.pck_at = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.pck_at = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 
                       0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                       0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
        self.pck_mat_all = np.zeros((self.n_hm, len(self.pck_at), self.len_test_set))  # 10, 16, 1000
        self.pck_mat_avg = np.zeros((self.n_hm, len(self.pck_at)))  # 10, 16
        self.pck_pixel_norm_len = np.zeros((self.len_test_set, len(self.pck_at)))  # n_test x len(pck_at)
        
        # for pckh
        # self.pckh_h_factor = 0.5  # commneted becasue it's included in the list below
#         self.pckh_at = [0.0, 0.02, 0.04, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#         self.pckh_at = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.pckh_at = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 
                       0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 
                       0.40, 0.42, 0.44, 0.46, 0.48, 0.50]
        self.pckh_mat_all = np.zeros((self.n_hm, len(self.pckh_at), self.len_test_set))  # 10, 16, 1000
        self.pckh_mat_avg = np.zeros((self.n_hm, len(self.pckh_at)))  # 10, 16
        self.pckh_pixel_norm_len = np.zeros((self.len_test_set, len(self.pckh_at)))  # n_test x len(pckh_at)
        
        # joint list
        self.joints = ['Head', 'Spine_Shoulder (Neck)', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Left Hip', 'Right Hip']
        self.joints_short = ['H', 'SpSh', 'LSh', 'LEl', 'LWr', 'RSh', 'REl', 'RWr', 'LHip', 'RHip']
        
        # CONFIG PARAMETERS
        self.param = None
        self.model_params = None
        
        
    def test_video(self, ):
        print("testing video", self.video_path, self.video_name, "in session", self.test_session_for_video)
        
        #1
        self.prepare_testset_and_load_model_for_video()
 
        # 2 config
        self.param, self.model_params = config_reader('config')
        print("self.param", self.param, type(self.param))
        print("self.model_params", self.model_params, type(self.model_params))
        ### TODO write and change config

        # param_dict = {'use_gpu': 1, 'GPUdeviceNumber': 0, 'modelID': '1', 'octave': 3, 'starting_range': 0.8, 'ending_range': 2.0, 'scale_search': [1, 1, 1], 'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5, 'min_num': 4, 'mid_num': 10, 'crop_ratio': 2.5, 'bbox_ratio': 0.25}
    
        # 3
        self.run_test_on_imgs()
        
        
    def prepare_testset_and_load_model_for_video(self,):
        
        print("\nself.n_stages", self.n_stages)
        print("self.branch_flag: [1 => branch L1 only; 2 => branch L2 only (heatmaps only)] ======", self.branch_flag)
        
        self.prepare_train_val_data_dict_object_based_version_for_videos()
        
        self.BASE_DIR_TEST_VIDEO = os.path.join(EXP_BASE_DIR, self.experiment_dir + "testing/videos/", self.video_folder_name)
        print("creating a directory", self.BASE_DIR_TEST_VIDEO)
        os.makedirs(self.BASE_DIR_TEST_VIDEO, exist_ok=True)
        
        self.BASE_DIR_TEST_KP = os.path.join(self.BASE_DIR_TEST_VIDEO, "kp")
        print("creating a directory", self.BASE_DIR_TEST_KP)
        os.makedirs(self.BASE_DIR_TEST_KP, exist_ok=True)

        self.BASE_DIR_TEST_HEATMAPS = os.path.join(self.BASE_DIR_TEST_VIDEO, "heatmaps")
        print("creating a directory", self.BASE_DIR_TEST_HEATMAPS)
        os.makedirs(self.BASE_DIR_TEST_HEATMAPS, exist_ok=True)
        
        self.BASE_DIR_TEST_RESULTS = os.path.join(self.BASE_DIR_TEST_VIDEO, "results")
        print("creating a directory", self.BASE_DIR_TEST_RESULTS)
        os.makedirs(self.BASE_DIR_TEST_RESULTS, exist_ok=True)
        
        self.BASE_DIR_TEST_IMAGES = os.path.join(self.BASE_DIR_TEST_VIDEO, "images")
        print("creating a directory", self.BASE_DIR_TEST_IMAGES)
        os.makedirs(self.BASE_DIR_TEST_IMAGES, exist_ok=True)
        

        
             
        if not rmpe_testing:
            # e.g., epoch_num = 100
            model_file = self.BASE_DIR_TRAIN + 'weights_egg.%04d.h5'%(int(self.epoch_num))
            
            self.model = get_testing_model_eggnog_v1(self.n_stages, self.branch_flag)
            self.model.load_weights(model_file)
            
        else:  # this is rmpe testing so load rmpe model and weights
            assert(self.branch_flag == 0)
            model_file = rmpe_weights_file
            
            self.model = get_rmpe_test_model(self.n_stages_rmpe)
            self.model.load_weights(model_file, by_name=True)
            print("Loaded RMPE model with n_stage = ", self.n_stages_rmpe)
        
        
    def prepare_train_val_data_dict_object_based_version_for_videos(self,):
        # 
        partition_test = []
        self.partition_dict['test'] = []

        ## create Session objects and withdraw images
        for test_s in self.test_session_for_video:
            print("\ntest_s ===============================", test_s)
            sess = Session(session_dir=os.path.join(eggnog_dataset_path, test_s), meta_dir=eggnog_meta_dir)
            sess.print_session_info()                
                
            partition_test = sess.get_all_the_frame_of_specific_video(self.video_session_name + "/" + self.video_layout_name + "/" + self.video_folder_name)

        print("After session objects draw: len(partition_test)", len(partition_test))

#         # shuffle test list
#         random.seed(random_seed)
#         random.shuffle(partition_test)

        # create test dict
        for i, img in enumerate(partition_test):    
            self.partition_dict['test'].append(img)
            if verbose: print("image i", i, img)
             
        self.total_test_samples = len(self.partition_dict['test'])
        print("\n##### final test_samples", len(self.partition_dict['test']))
        
        
        
    def test(self,):
        # 1
        self.prepare_testset_and_load_model()

        # 2 config
        self.param, self.model_params = config_reader('config')
        print("self.param", self.param, type(self.param))
        print("self.model_params", self.model_params, type(self.model_params))
        ### TODO write and change config

        # param_dict = {'use_gpu': 1, 'GPUdeviceNumber': 0, 'modelID': '1', 'octave': 3, 'starting_range': 0.8, 'ending_range': 2.0, 'scale_search': [1, 1, 1], 'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5, 'min_num': 4, 'mid_num': 10, 'crop_ratio': 2.5, 'bbox_ratio': 0.25}
    
        # 3
        self.run_test_on_imgs()
    
    
    def prepare_testset_and_load_model(self,):
        ##### prepare test dataset
        n_test_imgs_per_session = int(self.n_test_imgs/len(self.test_sessions))

        n_test_imgs_per_session_aug = int(self.aug_fraction*n_test_imgs_per_session)
        n_test_imgs_per_session_nonaug = int((1-self.aug_fraction)*n_test_imgs_per_session)

        print("self.test_sessions", self.test_sessions)
        print("self.n_test_imgs", self.n_test_imgs)
        print("n_test_imgs_per_session", n_test_imgs_per_session)

        print("aug_fraction", self.aug_fraction)
        print("n_test_imgs_per_session_aug", n_test_imgs_per_session_aug)
        print("n_test_imgs_per_session_nonaug", n_test_imgs_per_session_nonaug)

        self.prepare_train_val_data_dict_object_based_version(n_test_imgs_per_session_aug, n_test_imgs_per_session_nonaug)

        ##### prepare model
        print("\nself.n_stages", self.n_stages)
        print("self.branch_flag: [1 => branch L1 only; 2 => branch L2 only (heatmaps only)] ======", self.branch_flag)

        self.BASE_DIR_TEST_KP = os.path.join(EXP_BASE_DIR, self.experiment_dir + "testing/kp")
        print("creating a directory", self.BASE_DIR_TEST_KP)
        os.makedirs(self.BASE_DIR_TEST_KP, exist_ok=True)

        self.BASE_DIR_TEST_HEATMAPS = os.path.join(EXP_BASE_DIR, self.experiment_dir + "testing/heatmaps")
        print("creating a directory", self.BASE_DIR_TEST_HEATMAPS)
        os.makedirs(self.BASE_DIR_TEST_HEATMAPS, exist_ok=True)
        
        self.BASE_DIR_TEST_RESULTS = os.path.join(EXP_BASE_DIR, self.experiment_dir + "testing/results")
        print("creating a directory", self.BASE_DIR_TEST_RESULTS)
        os.makedirs(self.BASE_DIR_TEST_RESULTS, exist_ok=True)
        
        self.BASE_DIR_TEST_IMAGES = os.path.join(EXP_BASE_DIR, self.experiment_dir + "testing/images")
        print("creating a directory", self.BASE_DIR_TEST_IMAGES)
        os.makedirs(self.BASE_DIR_TEST_IMAGES, exist_ok=True)

        
        
        if not rmpe_testing:
            # e.g., epoch_num = 100
            model_file = self.BASE_DIR_TRAIN + 'weights_egg.%04d.h5'%(int(self.epoch_num))
            
            self.model = get_testing_model_eggnog_v1(self.n_stages, self.branch_flag)
            self.model.load_weights(model_file)
            
        else:  # this is rmpe testing so load rmpe model and weights
            assert(self.branch_flag == 0)
            model_file = rmpe_weights_file
            
            self.model = get_rmpe_test_model(self.n_stages_rmpe)
            self.model.load_weights(model_file, by_name=True)
            print("Loaded RMPE model with n_stage = ", self.n_stages_rmpe)
        
        
    
    def prepare_train_val_data_dict_object_based_version(self, n_test_imgs_per_session_aug, n_test_imgs_per_session_nonaug):
        # new version for eggnog dataset where images are drawn from the session objects
        partition_test = []
        self.partition_dict['test'] = []
        test_list_aug = []
        test_list_nonaug = []

        ## create Session objects and withdraw images
        for test_s in self.test_sessions:
            print("\ntest_s ===============================", test_s)
            sess = Session(session_dir=os.path.join(eggnog_dataset_path, test_s), meta_dir=eggnog_meta_dir)
            sess.print_session_info()

            # draw n_test_imgs_per_session_aug examples from v0
            if not n_test_imgs_per_session_aug == 0:
                test_list_aug = sess.get_evenly_spaced_n_images(n_imgs=n_test_imgs_per_session_aug, get_aug=True, aug_version="v0")

            # draw n_test_imgs_per_session_nonaug examples from non_aug set
            if not n_test_imgs_per_session_nonaug == 0:
                test_list_nonaug = sess.get_evenly_spaced_n_images(n_imgs=n_test_imgs_per_session_nonaug, get_aug=False, aug_version="")

            partition_test = partition_test + test_list_aug + test_list_nonaug

        print("After session objects draw: len(partition_test)", len(partition_test))

        #####
        if add_imagenet_images:
            # add random imagenet images without humans: imagenet_fraction % of len(partition_test)
            imagenet = ImagenetImages(imagenet_dir)
            imagenet_test = imagenet.get_n_images("test", int(imagenet_fraction*len(partition_test)))
            print("Before adding imagenet images:")
            print("len(partition_test)", len(partition_test))
            print("len(imagenet_test)", len(imagenet_test))

            # combine eggnog and imagenet lists
            partition_test = partition_test + imagenet_test
        #####

        # shuffle test list
        random.seed(random_seed)
        random.shuffle(partition_test)

        # create test dict
        for i, img in enumerate(partition_test):
            if i >= self.len_test_set:
                break
                
            self.partition_dict['test'].append(img)
            if verbose: print("image i", i, img)
            
        
        self.total_test_samples = len(self.partition_dict['test'])
        print("\n##### final test_samples", len(self.partition_dict['test']))

    
    def plot_pck_mat(self,):
        fig, ax = plt.subplots(nrows=5, ncols=2)
        print(ax.shape)
        fig.set_size_inches((5, 5))

        x = self.pck_at
        x_label = "Normalized Distance"
        y_label = "PCK"

        for p in range(self.pck_mat_avg.shape[0]):
            i = p%5
            j = 0 if p<5 else 1
            y = self.pck_mat_avg[p]
            ax[i][j].plot(x, y)
            ax[i][j].set(xlabel=x_label, ylabel=y_label, xlim=[0, 0.5], ylim=[0, 1.1], title=self.joints[p])
            ax[i][j].grid()

        # fig.tight_layout()
        plt.show()
        # fig.savefig(self.BASE_DIR_TEST_RESULTS + "/t1.png")
        
        
    def distAB(self, pointA, pointB):
        """
        pointA and B: should be np.array([x, y])
        """
        return np.linalg.norm(pointA - pointB)
    
    
#     def check_if_within_distance_d(self, pointA, pointB, d):
#         """
#         pointA: ground truth point (x, y)
#         pointB: predicted point (x, y)
#         d: self.pckh_h_factor*h (h in PCKh metric)
#         """
#         if pdist(np.array([pointA, pointB])) < d:
#             return True
#         else:
#             return False 
        
        
    def get_len_head_segment(self, head_xy, spine_shoulder_xy):
        len_h = self.distAB(np.array(head_xy), np.array(spine_shoulder_xy))
        return len_h
    
    
    def get_len_persons_bounding_box(self, gt_kp):
        # self.b_box_method = "vertical_box": this method assumes that test images and subjects are all upright there is no tilt in their spinal axis w.r.t. the vertical direction
        if self.b_box_method == "vertical_box":
            bw = max(gt_kp[:,0]) - min(gt_kp[:,0])  # max along x - min along x
            bh = max(gt_kp[:,1]) - min(gt_kp[:,1])  # max along y - min along y
        normalized_len = max(bh, bw)
        return normalized_len
    
    
    def print_and_save_pck_mats(self,):
        
        # print("self.pckh_mat_all\n", self.pckh_mat_all)
        print("self.pckh_mat_avg\n", self.pckh_mat_avg)
        print("saving pckh mat to", self.BASE_DIR_TEST_RESULTS)
        pckh_savefile = os.path.join(self.BASE_DIR_TEST_RESULTS, "test_pckh_ep" + str(self.epoch_num) + "_nTest_" + str(self.len_test_set) + "_time" + get_timestamp() + ".npy")
        np.save(pckh_savefile, self.pckh_mat_avg)
        print("mean pckh across all joints:", np.mean(self.pckh_mat_avg, axis=0))
        print("mean normalized len in pixels for every x for x in pckh@x\n", np.mean(self.pckh_pixel_norm_len, axis=0))
        
        print("\nself.pckh_at", self.pckh_at)
        
        # print("self.pck_mat_all\n", self.pck_mat_all)
        print("\nself.pck_mat_avg\n", self.pck_mat_avg)
        print("saving pck mat to", self.BASE_DIR_TEST_RESULTS)
        pck_savefile = os.path.join(self.BASE_DIR_TEST_RESULTS, "test_pck_ep" + str(self.epoch_num) + "_nTest_" + str(self.len_test_set) + "_time" + get_timestamp() + ".npy")
        np.save(pck_savefile, self.pck_mat_avg)
        print("mean pck across all joints:", np.mean(self.pck_mat_avg, axis=0))
        print("mean normalized len in pixels for every x for x in pck@x\n", np.mean(self.pck_pixel_norm_len, axis=0))

        print("\nself.pck_at", self.pck_at)
        
        
    def calculate_pck(self, idx, gt_kp, pred_kp):
        if verbose_pckh: 
            print(" >>>>>>>> gt_kp.shape, pred_kp.shape", gt_kp.shape, pred_kp.shape)
            print(" >>>>>>>>\n", gt_kp, "\n\n", pred_kp)
            print("checking if anything is nan in the ground truth")
        assert(not np.isnan(gt_kp).any())
        
        if rmpe_testing:
            pred_kp = pred_kp[rmpe_to_eggnog_slicing, :]  # rmpe predicted joint 11 is rhip which is joint index 9 in eggnog-common architecture, see sheet 2
            
            
        pckh_mat_img = np.zeros(self.pckh_mat_avg.shape)  # (10, 16)
        pck_mat_img = np.zeros(self.pck_mat_avg.shape)  # (10, 16)
        
        for metric in self.metric_type:
            if metric == "PCKh":
                # check if head and spine shoulder are not nan
                head_xy = gt_kp[0]
                spine_shoulder_xy = gt_kp[1]
                if verbose_pckh: print("gt head_xy, spine_shoulder_xy", head_xy, spine_shoulder_xy)
                assert(not np.isnan(head_xy).any())
                assert(not np.isnan(spine_shoulder_xy).any())
                
                len_head_seg = self.get_len_head_segment(head_xy, spine_shoulder_xy)
                assert(len_head_seg > 0)
                if verbose_pckh: print(" ***** PCKh h len_head_seg =", len_head_seg)
                
                for n_id, n in enumerate(self.pckh_at):
                    if verbose_pckh: print("calculating PCKh (w.r.t. head segment) at", n)
                    self.pckh_pixel_norm_len[idx][n_id] = n*len_head_seg
            
                    # comapare the gt and pred in pairwise manner and find the distances between pairs (n_hm pairs)
                    pckh_mat_img[:, n_id] = np.sqrt(np.sum(np.square(gt_kp - pred_kp[:, 0:2]), axis=1))
                    if verbose_pckh: print("pckh_mat_img.shape", pckh_mat_img.shape)  # (10, 16)
                    
                    # threshold based on whether the points are within 0.5*len_head_seg distance of each other
                    # if there is np.nan in predicted kp, then following d comparison is always False (as expected because the prediction is not close to gt), so it outputs 0 meaning the joint was not predicted correctly, as expected
                    pckh_mat_img[:, n_id] = np.array([1 if d <= n*len_head_seg else 0 for d in pckh_mat_img[:, n_id]])
                    # pckh_mat_img = pckh_mat_img[:, np.newaxis]  # from (10,) to (10,1)
                
                # set the calculated pckh_mat_img to the idx of self.pckh_mat_all
                # print("", self.pckh_mat_all[:, :, idx].shape, pckh_mat_img.shape)
                self.pckh_mat_all[:, :, idx] = pckh_mat_img
                if verbose_pckh: print("pckh_mat_img\n", pckh_mat_img)
                
            if metric == "PCK":
                normalized_len = self.get_len_persons_bounding_box(gt_kp)
                assert(normalized_len > 0)
                if verbose_pck: print(" ***** PCK normalized_len =", normalized_len)
                
                for m_id, m in enumerate(self.pck_at):
                    if verbose_pck: print("measuring pck at", m)
                    self.pck_pixel_norm_len[idx][m_id] = m*normalized_len
                    
                    # comapare the gt and pred in pairwise manner and find the distances between pairs (n_hm pairs)
                    pck_mat_img[:, m_id] = np.sqrt(np.sum(np.square(gt_kp - pred_kp[:, 0:2]), axis=1))
                    if verbose_pck: print("pck_mat_img, pck_mat_img[:, m_id].shape is (10,)", pck_mat_img, pck_mat_img[:, m_id].shape)  # to verify if the columns are exactly same to each other
                    
                    # threshold based on whether the points are within m*normalized_len distance of each other
                    pck_mat_img[:, m_id] = np.array([1 if d <= m*normalized_len else 0 for d in pck_mat_img[:, m_id]])
                    # pck_mat_img = pck_mat_img[:, np.newaxis]  # from (10,) to (10,1)
                    
                # set the calculated pck_mat_img to the idx of self.pck_mat_all
                if verbose_pck: print("self.pck_mat_all[:, :, idx].shape, pck_mat_img.shape", self.pck_mat_all[:, :, idx].shape, pck_mat_img.shape)
                self.pck_mat_all[:, :, idx] = pck_mat_img
                if verbose_pck: print("pck_mat_img\n", pck_mat_img)

                    
    def calculate_loss(self, test_image, heatmap_avg_pred):
        pred_hm = heatmap_avg_pred
        # print("pred shape", pred_hm.shape)
        
        if "240x320" in test_image:
            gt_path = test_image.split(".")[0].replace("240x320", "heatmap30x40.npy")
            gt_hm_19 = np.load(gt_path)
            gt_hm_10 = gt_hm_19[:,:,EggnogGlobalConfig.eggnog19_to_coco_10_mapping]
                    
            maxed_heatmap = np.max(gt_hm_10[:,:,:], axis=2)
            gt_hm_11 = np.dstack((gt_hm_10, 1 - maxed_heatmap))
                    
        else:
            # for imagenet images, use all black hm and all white background hm
            gt_hm_10 = np.zeros((pred_hm.shape[0], pred_hm.shape[1], pred_hm.shape[2]-1))
            
            maxed_heatmap = np.max(gt_hm_10[:,:,:], axis=2)
            gt_hm_11 = np.dstack((gt_hm_10, 1 - maxed_heatmap))

        # print("gt_hm_11 shape", gt_hm_11.shape)
                
        loss_hm = eucl_loss_np(pred_hm, gt_hm_11)
        print("loss_hm =============", loss_hm)
            
        return loss_hm
        
        
    def find_peaks_per_predicted_hm(self, heatmap_avg):
        all_peaks = []
        peak_counter = 0
        
        for part in range(np_branch2-1):
            map_ori = heatmap_avg[:,:,part]
            map = gaussian_filter(map_ori, sigma=3)

            map_left = np.zeros(map.shape)
            map_left[1:,:] = map[:-1,:]
            map_right = np.zeros(map.shape)
            map_right[:-1,:] = map[1:,:]
            map_up = np.zeros(map.shape)
            map_up[:,1:] = map[:,:-1]
            map_down = np.zeros(map.shape)
            map_down[:,:-1] = map[:,1:]

            peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map >  self.param['thre1']))
            # print("peaks_binary.shape", peaks_binary.shape)  # (480, 640)
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
            if verbose: print("peaks list", peaks)
            peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
            if verbose: print("peaks_with_score", peaks_with_score)
            
            
            # find the max score and only keep one peak (Need to remove this part when you add paf prediction network)
            if len(peaks_with_score) > 1:
                peaks_with_score = [max(peaks_with_score, key=itemgetter(2))]
                # peaks_with_score = [max(peaks_with_score, key=lambda pws:pws[2])]
                if verbose: print("peaks_with_score max score is kept", peaks_with_score)
            
            
            id = range(peak_counter, peak_counter + len(peaks_with_score))
            if verbose: print("id", id)
            peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]
            if verbose: print("peaks_with_score_and_id", peaks_with_score_and_id)
            
            # convert tuple to list: [(178, 146, 0.34160166233778, 9)] -> [178, 146, 0.34160166233778, 9]
            if len(peaks_with_score) > 0:
                peaks_with_score_and_id = list(peaks_with_score_and_id[0])
            else:  #  len(peaks_with_score) = 0:
                peaks_with_score_and_id = [np.nan, np.nan, np.nan, np.nan]
                
            all_peaks.append(peaks_with_score_and_id)
            # print("all_peaks", all_peaks)
            peak_counter += len(peaks_with_score)
    
        print("all_peaks, peak_counter", all_peaks, len(all_peaks), peak_counter)
        return all_peaks
    
    
    def show_overlapped_gt_and_pred_v1(self, test_image, heatmap_avg, gt_kp, pred_kp):
        # print("gt_kp, gt_kp.shape", gt_kp, gt_kp.shape)  # (10, 2)
        # print("pred_kp, pred_kp.shape", pred_kp, pred_kp.shape)  # (10, 2)
        
        if rmpe_testing:
            pred_kp = pred_kp[rmpe_to_eggnog_slicing, :]  # rmpe predicted joint 11 is rhip which is joint index 9 in eggnog-common architecture, see sheet 2
        
        MEDIUM_SIZE = 16
        plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
        
#         plt.axis('off')
        
        i = -1  # only background hm
        
        # =========================
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #fig.set_size_inches((5, 5))
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        #fig.add_axes(ax)
        #ax.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #extent = matplotlib.transforms.Bbox(((0.765, 0.493), (5.795, 4.2589999999999995)))
        #print("extent", extent)
        
        oriImg = cv2.imread(test_image)  # B,G,R order
        ax.imshow(oriImg[:,:,[2,1,0]], alpha=0.35)
        
        
        x = gt_kp[...,0]
        y = gt_kp[...,1]
        a1 = ax.scatter(x, y, c='b', alpha=0.75, s=15)
#         for n, txt in enumerate(self.joints_short):
#             ax.annotate(txt, (x[n],y[n]), fontsize=5, color='white')
        for p in paf_pairs_indices_10joints:
            plt.plot([gt_kp[p[0]][0], gt_kp[p[1]][0]], [gt_kp[p[0]][1], gt_kp[p[1]][1]], color='magenta', linestyle='-', lw=1)
        
        fig.savefig(os.path.join(self.BASE_DIR_TEST_IMAGES, test_image.split("/")[-1].split(".")[0] + '_gt.png'),  dpi=300, pad_inches=0, bbox_inches=extent)
        
        # =========================
        fig, ax = plt.subplots(nrows=1, ncols=1)
        #fig.set_size_inches((5, 5))
        ax.set_axis_off()
        #fig.add_axes(ax)
        #ax.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #print("extent", extent)
        
        oriImg = cv2.imread(test_image)  # B,G,R order
        ax.imshow(oriImg[:,:,[2,1,0]], alpha=0.35)
        
        x = pred_kp[...,0]
        y = pred_kp[...,1]
        a2 = ax.scatter(x, y, c='g', alpha=0.75, s=15)
#         for n, txt in enumerate(self.joints_short):
#             ax.annotate(txt, (x[n],y[n]), fontsize=5, color='darkslategrey')
        for p in paf_pairs_indices_10joints:
            plt.plot([pred_kp[p[0]][0], pred_kp[p[1]][0]], [pred_kp[p[0]][1], pred_kp[p[1]][1]], color='gold', linestyle='-', lw=1)
            
        fig.savefig(os.path.join(self.BASE_DIR_TEST_IMAGES, test_image.split("/")[-1].split(".")[0] + '_pred.png'),  dpi=300, bbox_inches=extent, pad_inches=0)
        
#         ax_h = ax.imshow(heatmap_avg[:,:,i], alpha=.70) 
#         ax.legend([a1, a2], ["GTruth", "Pred"])
#         fig.colorbar(ax_h, ax=ax)
        
#         fig.savefig(os.path.join(self.BASE_DIR_TEST_IMAGES, test_image.split("/")[-1].split(".")[0] + '.png'))
    
    
    def show_overlapped_gt_and_pred(self, test_image, heatmap_avg, gt_kp, pred_kp):
        if rmpe_testing:
            pred_kp = pred_kp[rmpe_to_eggnog_slicing, :]  # rmpe predicted joint 11 is rhip which is joint index 9 in eggnog-common architecture, see sheet 2
            
        i = -1  # only background hm
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches((5, 5))
        
        oriImg = cv2.imread(test_image)  # B,G,R order
        ax.imshow(oriImg[:,:,[2,1,0]])
        
        x = gt_kp[...,0]
        y = gt_kp[...,1]
        a1 = ax.scatter(x, y, c='w', alpha=0.5, s=10)
        for n, txt in enumerate(self.joints_short):
            ax.annotate(txt, (x[n],y[n]), fontsize=5, color='white')
        
        x = pred_kp[...,0]
        y = pred_kp[...,1]
        a2 = ax.scatter(x, y, c='k', alpha=0.5, marker='*', s=5)
        for n, txt in enumerate(self.joints_short):
            ax.annotate(txt, (x[n],y[n]), fontsize=5, color='darkslategrey')
        
        ax_h = ax.imshow(heatmap_avg[:,:,i], alpha=.70) 
        ax.legend([a1, a2], ["GTruth", "Pred"])
        fig.colorbar(ax_h, ax=ax)
        
        fig.savefig(os.path.join(self.BASE_DIR_TEST_IMAGES, test_image.split("/")[-1].split(".")[0] + '.png'))
    
    
        
    def process_img(self, idx, test_image):
        # print("\nprocessing img =========", test_image)
        # print("map listed", *self.param['scale_search'])
        # print("process img param", self.param)

        eggnog_h = EggnogGlobalConfig.height
        
        oriImg = cv2.imread(test_image)  # B,G,R order
        print("oriImage shape", oriImg.shape)
        
        if "240x320" not in test_image:
            if oriImg.shape[0] > EggnogGlobalConfig.height and oriImg.shape[1] > EggnogGlobalConfig.width:
                oriImg = oriImg[0:EggnogGlobalConfig.height, 0:EggnogGlobalConfig.width, :]
            else:
                oriImg = cv2.resize(oriImg, (EggnogGlobalConfig.width, EggnogGlobalConfig.height))
                
        heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], np_branch2))
        heatmap_avg_pred = np.zeros((int(EggnogGlobalConfig.height/EggnogGlobalConfig.ground_truth_factor), int(EggnogGlobalConfig.width/EggnogGlobalConfig.ground_truth_factor), np_branch2))
        
        if not eggnog_testing:
            multiplier = [x * self.model_params['boxsize'] / oriImg.shape[0] for x in self.param['scale_search']]
        else:
            multiplier = [x * eggnog_h / oriImg.shape[0] for x in self.param['scale_search']]
        # print("multiplier", multiplier)

        ######
        for m in range(len(multiplier)):
            #
            scale = multiplier[m]
            if verbose: print("scale ========", scale)
            imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if verbose: print("imageToTest.shape", imageToTest.shape)
            imageToTest_padded, pad = util.padRightDownCorner(imageToTest, self.model_params['stride'], self.model_params['padValue'])        
            if verbose: print("imageToTest_padded.shape", imageToTest_padded.shape)

            input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
            if verbose: print("Input shape: " + str(input_img.shape))  

            output_blobs = self.model.predict(input_img)
            if verbose: print("Output len: " + str(len(output_blobs)))

            #
            # extract outputs, resize, and remove padding
            if self.branch_flag == 2:
                heatmap = np.squeeze(output_blobs[0])  # output 0 is heatmaps for L2 only network
            elif self.branch_flag == 0:
                heatmap = np.squeeze(output_blobs[1])  # output 0 is heatmaps for L1 + L2 network
            else:
                raise NotImplementedError("L1 (paf) only version is not written yet.")
                
            if verbose: print("after squeeze heatmap", heatmap.shape)

#             # save to disk hm
#             if hm_save:
#                 print("BASE_DIR_TEST_HEATMAPS", self.BASE_DIR_TEST_HEATMAPS)
#                 hm_savefile = os.path.join(self.BASE_DIR_TEST_HEATMAPS, test_image.split("/")[-1].split(".")[0] + "_hm.npy") 
#                 np.save(hm_savefile, heatmap)
#                 print("saving", hm_savefile, "heatmap.shape", heatmap.shape)

            heatmap_avg_pred = heatmap_avg_pred + cv2.resize(heatmap, (0,0), fx=(1/scale), fy=(1/scale), interpolation=cv2.INTER_CUBIC) / len(multiplier)

            heatmap = cv2.resize(heatmap, (0,0), fx=self.model_params['stride'], fy=self.model_params['stride'], interpolation=cv2.INTER_CUBIC)
            if verbose: print("after resize heatmap", heatmap.shape)
            heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
            if verbose: print("after removing padding heatmap", heatmap.shape)
            heatmap = cv2.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

            # save to disk paf
        #     paf = np.squeeze(output_blobs[0]) # output 0 is PAFs
        #     paf = cv2.resize(paf, (0,0), fx=model_params['stride'], fy=model_params['stride'], interpolation=cv2.INTER_CUBIC)
        #     paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        #     paf = cv2.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            heatmap_avg = heatmap_avg + heatmap / len(multiplier)
            # paf_avg = paf_avg + paf / len(multiplier)
        
        if hm_save:
            # print("BASE_DIR_TEST_HEATMAPS", self.BASE_DIR_TEST_HEATMAPS)
            hm_savefile = os.path.join(self.BASE_DIR_TEST_HEATMAPS, test_image.split("/")[-1].split(".")[0] + "_hm.npy") 
            np.save(hm_savefile, heatmap_avg_pred)
            # print("saving", hm_savefile, "heatmap_avg_pred.shape", heatmap_avg_pred.shape)
        
        loss_hm = 0
        if calculate_loss:
            # pred_hm = np.load(hm_savefile)
            loss_hm = self.calculate_loss(test_image, heatmap_avg_pred)
            
        ### finding the actual kp locations
        all_peaks = self.find_peaks_per_predicted_hm(heatmap_avg)
        
        # round of pred to nearest pixel number
#         if roundof_pred:
#             all_peaks = np.around(all_peaks)
        
        # write the predicted kps to disk
        if kp_save:
            # print("BASE_DIR_TEST_KP", self.BASE_DIR_TEST_KP)
            kp_savefile = os.path.join(self.BASE_DIR_TEST_KP, test_image.split("/")[-1].split(".")[0] + "_kp.npy") 
            np.save(kp_savefile, all_peaks)

        ### pck calculations
        if "240x320" in test_image:
            gt_kp_npy = test_image.split("_240x320")[0] + '.npy'
            kpi = np.load(gt_kp_npy)  # gt_kp = np.reshape(gt_kp, (len(gt_kp)/3, 3))
            gt_kp = np.delete(kpi, np.arange(0, kpi.size, 3))
            gt_kp_240x320 = keypoint_transform_to_240x320_image(gt_kp, flip=False)
            # print("gt kp", gt_kp.shape)
            # print("gt kp 240x320", gt_kp_240x320.shape)
            
            # convert (38,) to (19,2)
            gt_kp_240x320 = np.reshape(gt_kp_240x320, (int(len(gt_kp_240x320)/2), 2))
            # print("gt kp 240x320", gt_kp_240x320.shape)
           
            # select coco 10 joints
            gt_kp_240x320_10joints = gt_kp_240x320[EggnogGlobalConfig.eggnog19_to_coco_10_mapping]
            # print("gt kp 240x320 10 joints", gt_kp_240x320_10joints.shape)
            
        else:
            gt_kp_240x320_10joints = np.full((np_branch2-1, 2), np.nan)
            # print("gt kp 240x320_10joints with all np.nan", gt_kp_240x320_10joints.shape)
            
        self.calculate_pck(idx, gt_kp=gt_kp_240x320_10joints, pred_kp=np.array(all_peaks))
        
        # show and save overlapped gt and pred
        if img_save:
            self.show_overlapped_gt_and_pred_v1(test_image, heatmap_avg, gt_kp=gt_kp_240x320_10joints, pred_kp=np.array(all_peaks)[:, 0:2])
        
        return loss_hm
                
                                      
    def run_test_on_imgs(self,):
        total_loss = 0
        print("len self.partition_dict['test']", len(self.partition_dict['test']))
        for idx, img in enumerate(self.partition_dict['test']):
            if "train_set" in img:
                image = os.path.join(imagenet_dir, img)
            else:
                image = os.path.join(eggnog_dataset_path, img + '_240x320.jpg')

            print("\nreading and processing", idx, image)
            total_loss = total_loss + self.process_img(idx, image)
        
        self.total_loss = total_loss
        print("\nself.total_loss, self.total_test_samples", self.total_loss, self.total_test_samples)
        print("total_loss average", self.total_loss/self.total_test_samples)
        
        
        # find average across all test examples
        print("###############")
        print("Checking if *_mat_all has any np.nan in it (there should not be)")
        print("number of np.nans in self.pckh_mat_all", np.count_nonzero(np.isnan(self.pckh_mat_all)))
        print("np.isnan(self.pckh_mat_all).any() (should be false)", np.isnan(self.pckh_mat_all).any())
        
        print("number of np.nans in self.pck_mat_all", np.count_nonzero(np.isnan(self.pck_mat_all)))
        print("np.isnan(self.pck_mat_all).any() (should be false)", np.isnan(self.pck_mat_all).any())
        print("###############")
        
        self.pckh_mat_avg = np.mean(self.pckh_mat_all, axis=2)
        self.pck_mat_avg = np.mean(self.pck_mat_all, axis=2)
        
        # finally print and save
        self.print_and_save_pck_mats()
        # self.plot_pck_mat() use ipython to plot, this function doesn't plot good
        
        
        
if __name__ == "__main__":
    """
    Usage: provide exp_dir as argv[1], sys.argv[2] e.g., python test_pose.py common_train/0706180200pm/ 50
    For rmpe_test = True python test_pose.py rmpe_test/test_320x240_v2/ 0
    python test_pose.py exp1_v1/1002180300pm/ 100
    python test_pose_videos.py exp1_v1/1005180500pm/ 100 /s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s18/part1_layouts_p36/20160129_192616_00_Video.avi

    """
    
    if len(sys.argv) != 3 or len(sys.argv) != 4:
        print("Usage python test_pose.py exp1_v1/1002180300pm/ 100 OR test_pose.py exp1_v1/1002180300pm/ 100 /path/to/video")
    
    if len(sys.argv) == 3:
        test = Test(sys.argv[1], sys.argv[2], None)
        test.test()
        print("Testing done!")
        
    if len(sys.argv) == 4:
        test = Test(sys.argv[1], sys.argv[2], sys.argv[3])
        test.test_video()
        print("Testing done for video!")
