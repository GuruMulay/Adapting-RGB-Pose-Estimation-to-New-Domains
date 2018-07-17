import sys
import os
import numpy as np
sys.path.append("..")
from glob import glob

from model import get_testing_model_eggnog_v1

# 
import cv2
import matplotlib
import pylab as plt

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

np_branch1 = 18  # 18 (keeping only common joints and paf pairs) # 38
np_branch2 = 11  # 11 (keeping only common joints and paf pairs) # 19


os.environ["CUDA_VISIBLE_DEVICES"]="0"

EXP_BASE_DIR = "/s/red/b/nobackup/data/eggnog_cpm/training_files/"
eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/"

imagenet_dir = '/s/red/a/nobackup/imagenet/images/train/'
eggnog_meta_dir = '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_meta/'


add_imagenet_images = True
imagenet_fraction = 0.1

eggnog_testing = True

calculate_loss = True
hm_save = False  # save predicted hms to disk
kp_save = True

verbose = False

def eucl_loss_np(x, y):
    l = np.sum(np.square(x - y))  
    return l


class Test:
    
    def __init__(self, experiment_dir, epoch_num):
        
        self.experiment_dir = experiment_dir
        self.epoch_num = epoch_num
        
        self.BASE_DIR_TRAIN = os.path.join(EXP_BASE_DIR, self.experiment_dir + "training/weights_egg/")
        
        self.BASE_DIR_TEST_KP = ""
        self.BASE_DIR_TEST_HEATMAPS = ""
        
        # stores test data
        self.partition_dict = {}
        
        self.model = None
        
        # NETWORK PARAMAETERS
        self.n_stages = 2
        self.branch_flag = 2  # 0 => both branches; 1 => branch L1 only; 2 => branch L2 only (heatmaps only)
        
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
        self.n_test_imgs = 5000
        self.len_test_set = 10
        self.aug_fraction = 0.0  # use aug_fraction % of the images from aug set and remaining from original non_aug set
                
        # loss calc
        self.total_loss = 0
        self.total_test_samples = 0 
        
        # pck calculations
        self.pck_at = [0.1, 0.2, 0.3]  # @_, @_, and @_
        self.n_hm = 10  # 10 joints
        self.pck_mat = np.zeros((self.n_hm, len(self.pck_at)))
        
        
        # CONFIG PARAMETERS
        self.param = None
        self.model_params = None
        
        
    def test(self,):
        # 1
        self.prepare_testset_and_load_model()

        # 2 config
        self.param, self.model_params = config_reader('config')
        print("self.param", self.param, type(self.param))
        print("self.model_params", self.model_params, type(self.model_params))
        ### TODO write and change config

        #     param_dict = {'use_gpu': 1, 'GPUdeviceNumber': 0, 'modelID': '1', 'octave': 3, 'starting_range': 0.8, 'ending_range': 2.0, 'scale_search': [1, 1, 1], 'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5, 'min_num': 4, 'mid_num': 10, 'crop_ratio': 2.5, 'bbox_ratio': 0.25}
    
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

        # e.g., epoch_num = 100
        model_file = self.BASE_DIR_TRAIN + 'weights_egg.%04d.h5'%(int(self.epoch_num))

        self.model = get_testing_model_eggnog_v1(self.n_stages, self.branch_flag)
        self.model.load_weights(model_file)
    
    
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
        random.seed(115)
        random.shuffle(partition_test)

        # create test dict
        for i, img in enumerate(partition_test):
            self.partition_dict['test'].append(img)
            if verbose: print("", i, img)
            if i >= self.len_test_set:
                break
        
        self.total_test_samples = len(self.partition_dict['test'])
        print("\n##### final test_samples", len(self.partition_dict['test']))

    
    def calculate_pck(self, gt_kp, pred_kp):
        print(" >>>>>>>> gt_kp.shape, pred_kp.shape", gt_kp.shape, pred_kp.shape)
        pck_mat_img = np.zeros(self.pck_mat.shape)
        
        for m in self.pck_at:
            print("measuring pck at", m)
        
        return pck_mat_img
    
        
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
                if verbose: print("peaks_with_score", peaks_with_score)
            
            
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
    
        
    def process_img(self, test_image):
        print("\nprocessing img =========", test_image)
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
        heatmap_avg_pred = np.zeros((int(EggnogGlobalConfig.height/EggnogGlobalConfig.ground_truth_factor), int(EggnogGlobalConfig.width/EggnogGlobalConfig.ground_truth_factor), 11))
        
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
            heatmap = np.squeeze(output_blobs[0])  # output 0 is heatmaps for L2 only network
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
                
        if calculate_loss:
            # pred_hm = np.load(hm_savefile)
            loss_hm = self.calculate_loss(test_image, heatmap_avg_pred)
            
        ### finding the actual kp locations
        all_peaks = self.find_peaks_per_predicted_hm(heatmap_avg)
        
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
            print("gt kp", gt_kp, gt_kp.shape)
            print("gt kp 240x320", gt_kp_240x320.shape)
            
            # convert (38,) to (19,2)
            gt_kp_240x320 = np.reshape(gt_kp_240x320, (int(len(gt_kp_240x320)/2), 2))
            print("gt kp 240x320", gt_kp_240x320.shape)
           
            # select coco 10 joints
            gt_kp_240x320_10joints = gt_kp_240x320[EggnogGlobalConfig.eggnog19_to_coco_10_mapping]
            print("gt kp 240x320 10joints", gt_kp_240x320_10joints.shape)
            
        else:
            gt_kp_240x320_10joints = np.full((np_branch2-1, 2), np.nan)
            print("gt kp 240x320_10joints with all np.nan", gt_kp_240x320_10joints.shape)
            
        pck_mat_img = self.calculate_pck(gt_kp=gt_kp_240x320_10joints, pred_kp=all_peaks)
        
        
        return loss_hm
                
                                      
    def run_test_on_imgs(self,):
        total_loss = 0
        for idx, img in enumerate(self.partition_dict['test']):
            if "train_set" in img:
                image = os.path.join(imagenet_dir, img)
            else:
                image = os.path.join(eggnog_dataset_path, img + '_240x320.jpg')

            # print("reading", idx, image)
            total_loss = total_loss + self.process_img(image)
        
        self.total_loss = total_loss
        print("\nself.total_loss, self.total_test_samples", self.total_loss, self.total_test_samples)
        print("total_loss average", self.total_loss/self.total_test_samples)
                                      
        
        
if __name__ == "__main__":
    """
    Usage: provide exp_dir as argv[1], sys.argv[2] e.g., python test_pose.py common_train/0706180200pm/ 50
    """
    test = Test(sys.argv[1], sys.argv[2])
    test.test()
    print("Testing done!")