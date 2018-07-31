import os
import sys
import numpy as np
import random
import pprint
from natsort import natsorted
import pickle
import csv


# imagenet_folders_without_human = ['train_set_1/n01443537/', 'train_set_1/n01580077/', 'train_set_2/n02226429/', 'train_set_5/n13044778/', 'train_set_2/n02268443/', 'train_set_1/n01775062/', 'train_set_1/n01608432/']

# imagenet_images_with_rare_human = ['train_set_1/n01530575/', 'train_set_3/n03530642/', 'train_set_4/n03717622/', 'train_set_4/n03777754/', 'train_set_5/n07715103/']


# imagenet_folders_all = imagenet_folders_without_human  + imagenet_images_with_rare_human
# random.shuffle(imagenet_folders_all)

# imagenet_train = imagenet_folders_all[:int(0.8*len(imagenet_folders_all))]
# imagenet_val = imagenet_folders_all[int(0.8*len(imagenet_folders_all)):]

# print("train and val", imagenet_train, imagenet_val)


# 
n_data_folders_per_layout = 5
n_aug_images_per_frame = 5
div_factor_aug = 5  # with 5 only _aug_0 is selected
eggnog_meta_dir = '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_meta/'
verbose = False


class Session:

    def __init__(self, session_dir='', meta_dir=''):

        self.session_dir = session_dir.rstrip(os.sep)
        self.session_name = self.session_dir.split('/')[-1]
        self.meta_dir = os.path.join(meta_dir, self.session_name) 
        os.makedirs(self.meta_dir, exist_ok=True)
        
        # layouts
        self.layouts = natsorted([l for l in os.listdir(self.session_dir) if "layout" in l])
        assert(len(self.layouts) == 2)
        
        # assuming there will be only two people in one session
        # for layout 0 (person A)
        self.layoutA_dict = {}  # non-augmented data dict for person A
        self.layoutA_aug0_dict = {}  # augmented data dict for person A
        self.layoutA_aug1_dict = {}  # augmented data dict for person A
        # for layout 1 (person B)
        self.layoutB_dict = {}  # non-augmented data dict for person B
        self.layoutB_aug0_dict = {}  # augmented data dict for person B
        self.layoutB_aug1_dict = {}  # augmented data dict for person B
        
        self.dict_list = ["layoutA_dict", "layoutA_aug0_dict", "layoutA_aug1_dict", "layoutB_dict", "layoutB_aug0_dict", "layoutB_aug1_dict"]
        
        # 
        self.populate_session_info()
        
    
    
    def print_layout_dict_stats(self, layout_dict, dict_name):
        print("\nLayout dict: {}".format(dict_name))
        v_list = [k for k in layout_dict.keys()]
        f_count_list = [len(layout_dict[k]) for k in layout_dict.keys()]
        print("Videos:", v_list)
        print("nImages:", f_count_list)
        self.write_session_stats(v_list, f_count_list, write_dir=self.meta_dir, file_name=dict_name.split('.')[1])
        
        
    def print_session_info(self,):
        print("\nSession {} info ====================================================".format(self.session_name))
        print("Session Dir: {}".format(self.session_dir))
        print("Metadata Dir: {}".format(self.meta_dir))
        print("Layouts:", self.layouts)
        
        """
        print("Layout A dicts:")
        pprint.pprint(self.layoutA_dict)
        pprint.pprint(self.layoutA_aug0_dict)
        pprint.pprint(self.layoutA_aug1_dict)
        
        print("Layout B dicts:")
        pprint.pprint(self.layoutB_dict)
        pprint.pprint(self.layoutB_aug0_dict)
        pprint.pprint(self.layoutB_aug1_dict)
        """
        
        if verbose:
            self.print_layout_dict_stats(self.layoutA_dict, "self.layoutA_dict")
            self.print_layout_dict_stats(self.layoutA_aug0_dict, "self.layoutA_aug0_dict")
            self.print_layout_dict_stats(self.layoutA_aug1_dict, "self.layoutA_aug1_dict")

            self.print_layout_dict_stats(self.layoutB_dict, "self.layoutB_dict")
            self.print_layout_dict_stats(self.layoutB_aug0_dict, "self.layoutB_aug0_dict")
            self.print_layout_dict_stats(self.layoutB_aug1_dict, "self.layoutB_aug1_dict")

    
    def populate_session_info(self,):
        # populate only if not populated already
        # check if dicts are present already 
        dict_exists = [os.path.isfile(os.path.join(self.meta_dir, f + ".pkl")) for f in self.dict_list]
        if verbose:
            print("Does dicts exist", self.dict_list, dict_exists, all(dict_exists))
        if all(dict_exists):
            self.layoutA_dict = self.load_dict_pickle("layoutA_dict")
            self.layoutA_aug0_dict = self.load_dict_pickle("layoutA_aug0_dict")
            self.layoutA_aug1_dict = self.load_dict_pickle("layoutA_aug1_dict")
            self.layoutB_dict = self.load_dict_pickle("layoutB_dict")
            self.layoutB_aug0_dict = self.load_dict_pickle("layoutB_aug0_dict")
            self.layoutB_aug1_dict = self.load_dict_pickle("layoutB_aug1_dict")
            # print("Loaded dicts from the meta dir")
        else:
            # populate the dicts
            for ln, layout in enumerate(self.layouts):
                video_folders = [vf for vf in os.listdir(os.path.join(self.session_dir, layout)) if os.path.isdir(os.path.join(self.session_dir, layout, vf))]
                videos = [vf for vf in os.listdir(os.path.join(self.session_dir, layout)) if vf.endswith(".avi")]
                print("video folders and videos", video_folders, videos)
                print("layout and len video folders and videos", layout, len(video_folders), len(videos))

                assert(len(video_folders) == n_data_folders_per_layout*len(videos))

                for video_folder in video_folders:

                    # if "version" not in video_folder and "augmented" not in video_folder:
                    if "version" not in video_folder and "augmented" not in video_folder and "Aug" not in video_folder:
                        print("vf non-augmented ============= ", video_folder)
                        img_files = natsorted([im.split("_240x320")[0] for im in os.listdir(os.path.join(self.session_dir, layout, video_folder)) if im.endswith(".jpg") and "240x320" in im])

                        # update dict
                        if ln == 0:
                            self.layoutA_dict[os.path.join(self.session_name, layout, video_folder)] = img_files
                        else:
                            self.layoutB_dict[os.path.join(self.session_name, layout, video_folder)] = img_files

                        # for file in img_files:
                           # print(file)

                    # elif "augmented_v1" in video_folder:
                    elif "Aug_v1" in video_folder:
                        # print("vf augmented_v1 ============= ", video_folder)
                        print("vf Aug_v1 ============= ", video_folder)
                        img_files = natsorted([im.split("_240x320")[0] for im in os.listdir(os.path.join(self.session_dir, layout, video_folder)) if im.endswith(".jpg") and "240x320" in im])

                        # update dict
                        if ln == 0:
                            self.layoutA_aug1_dict[os.path.join(self.session_name, layout, video_folder)] = img_files
                        else:
                            self.layoutB_aug1_dict[os.path.join(self.session_name, layout, video_folder)] = img_files

                    # elif "augmented" in video_folder:
                    elif "Aug_v0" in video_folder:
                        # print("vf augmented ============= ", video_folder)
                        print("vf Aug_v0 ============= ", video_folder)
                        img_files = natsorted([im.split("_240x320")[0] for im in os.listdir(os.path.join(self.session_dir, layout, video_folder)) if im.endswith(".jpg") and "240x320" in im])

                        # update dict
                        if ln == 0:
                            self.layoutA_aug0_dict[os.path.join(self.session_name, layout, video_folder)] = img_files
                        else:
                            self.layoutB_aug0_dict[os.path.join(self.session_name, layout, video_folder)] = img_files

            print("Populated the layout dicts. Now saving ...")
            self.save_pickled_dictionaries()
    

    def save_dict_pickle(self, obj, name):
        with open(os.path.join(self.meta_dir, name + '.pkl'), 'wb') as f:
            print("writing {} ...".format(os.path.join(self.meta_dir, name + '.pkl')))
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

            
    def load_dict_pickle(self, name):
        with open(os.path.join(self.meta_dir, name + '.pkl'), 'rb') as f:
            if verbose:
                print("loading {} ...".format(os.path.join(self.meta_dir, name + '.pkl')))
            return pickle.load(f)
    
    
    def save_pickled_dictionaries(self, ):
        print("\nsaving all 6 layout dict pickle files to {}...".format(self.meta_dir))
        
        self.save_dict_pickle(self.layoutA_dict, "layoutA_dict")
        self.save_dict_pickle(self.layoutA_aug0_dict, "layoutA_aug0_dict")
        self.save_dict_pickle(self.layoutA_aug1_dict, "layoutA_aug1_dict")
        
        self.save_dict_pickle(self.layoutB_dict, "layoutB_dict")
        self.save_dict_pickle(self.layoutB_aug0_dict, "layoutB_aug0_dict")
        self.save_dict_pickle(self.layoutB_aug1_dict, "layoutB_aug1_dict")
        
        
    def write_session_stats(self, l1, l2, write_dir, file_name):
        with open(os.path.join(self.meta_dir, self.session_name + file_name + '.csv'), 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(l1, l2))
        
    
    def __get_image_list_from_dict_non_aug(self, layout_dict, div_factor):
        frame_list = []
        for v in layout_dict.keys():
            for f in layout_dict[v]:  # example: 20151114_022008_00_Video_vfr_2_skfr_1
                if int(f.split("_")[-3])%div_factor == 0:
                    frame_list.append(os.path.join(v, f))
                    # print("", os.path.join(v, f))
         
        return frame_list
    
    
    def __get_image_list_from_dict_aug(self, layout_dict, div_factor, div_factor_aug):
        frame_list = []
        for v in layout_dict.keys():
            for f in layout_dict[v]:  # example: 20151203_234151_00_Video_vfr_0_skfr_0_aug_0
                if int(f.split("_")[-5])%div_factor == 0 and int(f.split("_")[-1])%div_factor_aug == 0:
                    frame_list.append(os.path.join(v, f))
                    # print("added ", os.path.join(v, f))
         
        return frame_list
        
        
    def get_evenly_spaced_n_images(self, n_imgs, get_aug=False, aug_version=""):
        """
        get_aug=False means that the sample are taken from the non-augmented set of this session.
        returns n images that are evenly distributed among the nv videos and within every video the image samples are evenly spaced temporally.
        """
        imgs_per_layout = int(n_imgs/2)
        
        if not get_aug:
            print("Reading non-augmented folders ...")
            # layout A
            n_videos_A = len(self.layoutA_dict.keys())
            total_frames_A = np.sum([len(self.layoutA_dict[k]) for k in self.layoutA_dict.keys()])
            div_factor_A = int(total_frames_A/imgs_per_layout)
            print("\nlayoutA: div_factor", div_factor_A)
            print("layoutA: distributing {} images into {} videos".format(imgs_per_layout, n_videos_A))
            print("total number of frames from all videos of layout A =", total_frames_A)
            
            frame_list_A = self.__get_image_list_from_dict_non_aug(self.layoutA_dict, div_factor_A)
            print("Added n elements from layout A", len(frame_list_A))
            assert(len(frame_list_A) >= imgs_per_layout)
            
            # layout B
            n_videos_B = len(self.layoutB_dict.keys())
            total_frames_B = np.sum([len(self.layoutB_dict[k]) for k in self.layoutB_dict.keys()])
            div_factor_B = int(total_frames_B/imgs_per_layout)
            print("\nlayoutB: div_factor", div_factor_B)
            print("layoutB: distributing {} images into {} videos".format(imgs_per_layout, n_videos_B))
            print("total number of frames from all videos of layout B =", total_frames_B)
            
            frame_list_B = self.__get_image_list_from_dict_non_aug(self.layoutB_dict, div_factor_B)
            print("Added n elements from layout B", len(frame_list_B))
            assert(len(frame_list_B) >= imgs_per_layout)
            
            # layout A + layout B
            ret_frame_list = frame_list_A[0:imgs_per_layout] + frame_list_B[0:imgs_per_layout]  # crop the list to the limit of imgs_per_layout
            print("\nlen(ret_frame_list) =======", len(ret_frame_list))
            
            return ret_frame_list
        
        else:
            if aug_version == "v0":
                layoutA_aug_dict = self.layoutA_aug0_dict
                layoutB_aug_dict = self.layoutB_aug0_dict
            elif aug_version == "v1":
                layoutA_aug_dict = self.layoutA_aug1_dict
                layoutB_aug_dict = self.layoutB_aug1_dict
            
            print("Reading augmented version {} folders ...".format(aug_version))
            # layout A aug0
            n_videos_A = len(layoutA_aug_dict.keys())
            total_frames_A = np.sum([len(layoutA_aug_dict[k]) for k in layoutA_aug_dict.keys()])/n_aug_images_per_frame  # the actual sum number is bloated n_aug_images_per_frame times; hence divide by n_aug_images_per_frame
            div_factor_A = int(total_frames_A/imgs_per_layout)
            div_factor_aug_A = div_factor_aug
            
            print("\nlayoutA: div_factor", div_factor_A)
            print("layoutA: distributing {} images into {} videos".format(imgs_per_layout, n_videos_A))
            print("total number of frames from all videos of layout A _aug {} =".format(aug_version), total_frames_A)
            
            frame_list_A = self.__get_image_list_from_dict_aug(layoutA_aug_dict, div_factor_A, div_factor_aug_A)
            print("Added n elements from layout A", len(frame_list_A))
            assert(len(frame_list_A) >= imgs_per_layout)
            
            # layout B aug0
            n_videos_B = len(layoutB_aug_dict.keys())
            total_frames_B = np.sum([len(layoutB_aug_dict[k]) for k in layoutB_aug_dict.keys()])/n_aug_images_per_frame
            div_factor_B = int(total_frames_B/imgs_per_layout)
            div_factor_aug_B = div_factor_aug
            
            print("\nlayoutB: div_factor", div_factor_B)
            print("layoutB: distributing {} images into {} videos".format(imgs_per_layout, n_videos_B))
            print("total number of frames from all videos of layout B _aug {} =".format(aug_version), total_frames_B)
            
            frame_list_B = self.__get_image_list_from_dict_aug(layoutB_aug_dict, div_factor_B, div_factor_aug_B)
            print("Added n elements from layout B", len(frame_list_B))
            assert(len(frame_list_B) >= imgs_per_layout)
            
            ret_frame_list = frame_list_A[0:imgs_per_layout] + frame_list_B[0:imgs_per_layout]  # crop the list to the limit of imgs_per_layout
            print("\nlen(ret_frame_list) =======", len(ret_frame_list))
            
            return ret_frame_list
        
        
# def test_imagenet_image_reader():
#     im = ImagenetImages('/s/red/a/nobackup/imagenet/images/train/')
#     train = im.get_n_images("train", 100)
#     val = im.get_n_images("val", 100)
#     # print("train and val", train, val)
 

def test_session_general():
    s05 = Session('/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s20/', eggnog_meta_dir)
    s05.print_session_info()
    train_list = s05.get_evenly_spaced_n_images(n_imgs=1000, get_aug=True, aug_version="v0")
    print("assert if all unique", np.unique(train_list).size == len(train_list))
    print("train_list", len(train_list))  #, train_list)
    
    pprint.pprint(train_list)
    
    
def process_session(sess_dir):
    sess = Session(sess_dir, eggnog_meta_dir)
    sess.print_session_info()
    
    # test by drawing 1000 examples from v0
    train_list = sess.get_evenly_spaced_n_images(n_imgs=1000, get_aug=True, aug_version="v0")
    print("assert if all unique", np.unique(train_list).size == len(train_list))
    print("train_list", len(train_list))
#     pprint.pprint(train_list)
    

if __name__ == "__main__":
    # test_session_general()
    """
    Usage: provide session_dir as argv[1]
    """
    process_session(sys.argv[1])
    