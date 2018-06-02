import sys
import os
import re
import math
sys.path.append("..")

from model import get_training_model_eggnog
from dataset_gen import DataGenerator  #g
from optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19
import keras.backend as K

import random



verbose_print = True

split_videowise = False  # split train and val for the same video, 70% frames for train and 30% frames for val
split_sessionwise = True  # e.g., s04 for training s07 for validation; OR split train and val sessionwise, 70% session for train and 30% session for val


# sessionwise split
if split_sessionwise:
    train_sessions = ['s05']  # , 's02', 's03', 's04']
#     val_sessions = ['s05']
     
    # only take 1/div_factor fraction of data
    div_factor_train = 1
#     div_factor_val = 20
    
    print("train_sessions", train_sessions)
    print("div_factor_train", div_factor_train)
    

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


# eggonog sessions
# eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/"  # original size dataset
eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/"  # small dataset
print("eggnog_dataset_path ==============", eggnog_dataset_path)

params = {'data_path': eggnog_dataset_path,
          'height': 240,
          'width': 320,
          'n_channels': 3,
          'batch_size': batch_size,
          'paf_height': 30,
          'paf_width': 40,
          'paf_n_channels': 36,
          'hm_height': 30,
          'hm_width': 40,
          'hm_n_channels': 20,
          'save_transformed_path': None
         }
# '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/transformed/r2/'
# '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/transformed/r1/'




# =====
# new version for eggnog dataset where images are distributed in sessions
partition_train = []
partition_val = []
partition_dict = {}
partition_dict['train'] = []
partition_dict['val'] = []


if split_sessionwise:   
    # go through sessions and add img path to lists
    for session_name in train_sessions:
        for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf)) and "version" not in os.path.join(eggnog_dataset_path, session_name, layout, vf)]:
                print("train video_folder =====================", os.path.join(session_name, layout, video_folder))

                for file in sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder))):
                    if file.endswith('.jpg') and "240x320" in file:
                        if int(file.split('_')[-4])%div_factor_train == 0:  # append only if vfr number divisible by div_factor
                            # print(file)
                            partition_train.append(session_name + "/" + layout + "/" +  video_folder + "/" + file[:-12])  # append the path from base dir = eggnog_dataset_dir

    # create val list
    for session_name in val_sessions:
        for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf)) and "version" not in os.path.join(eggnog_dataset_path, session_name, layout, vf)]:
                print("val video_folder =====================", os.path.join(session_name, layout, video_folder))

                for file in sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder))):
                    if file.endswith('.jpg') and "240x320" in file:
                        if int(file.split('_')[-4])%div_factor_val == 0:  # append only if vfr number divisible by div_factor
                            # print(file)
                            partition_val.append(session_name + "/" + layout + "/" +  video_folder + "/" + file[:-12])  # append the path from base dir = eggnog_dataset_dir

                            
if split_videowise:
    # go through sessions and add img path to lists
    # create train list and val list simultaneously
    for session_name in train_val_sessions:
        for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf)) and "version" not in os.path.join(eggnog_dataset_path, session_name, layout, vf)]:
                files_list_video_folder = sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder)))
                n_files_in_video_folder = len(files_list_video_folder)  # includes .jpg and .npy*3 (4 files per image)
                print("train and val video_folder, n_files =====================", os.path.join(session_name, layout, video_folder), n_files_in_video_folder)
                
                # train append for first 70% frames
                for file in files_list_video_folder[0:int(1*n_files_in_video_folder)]:  # [0:int(0.7*n_files_in_video_folder)]:
                    if file.endswith('.jpg') and "240x320" in file:
                        # if int(file.split('_')[-4])%div_factor_train_val == 0:  # append only if vfr number divisible by div_factor
                        if int(file.split('_')[-4])%div_factor_train_val == 0 and int(file.split('_')[-4])%2 == 0:  # append only if vfr number divisible by div_factor and even
                            # print(file)
                            partition_train.append(session_name + "/" + layout + "/" +  video_folder + "/" + file[:-12])  # append the path from base dir = eggnog_dataset_dir
                            
                # val append for last 30% frames
                for file in files_list_video_folder[int(0*n_files_in_video_folder):]:  # [int(0.7*n_files_in_video_folder):]
                    if file.endswith('.jpg') and "240x320" in file:
                        # if int(file.split('_')[-4])%div_factor_train_val == 0:  # append only if vfr number divisible by div_factor
                        if int(file.split('_')[-4])%div_factor_train_val == 0 and int(file.split('_')[-4])%2 != 0:  # append only if vfr number divisible by div_factor and odd
                            # print(file)
                            partition_val.append(session_name + "/" + layout + "/" +  video_folder + "/" + file[:-12])  # append the path from base dir = eggnog_dataset_dir
                
                print("first 70% and last 30% len", len(files_list_video_folder[0:int(0.7*n_files_in_video_folder)]),  len(files_list_video_folder[int(0.7*n_files_in_video_folder):]))
                
                # for 100:100 split 0327180400pm
                print("first 100% and last 100% len", len(files_list_video_folder[0:int(1*n_files_in_video_folder)]),  len(files_list_video_folder[int(0*n_files_in_video_folder):]))

    
# shuffle train and val list
random.seed(115)
random.shuffle(partition_train)
random.shuffle(partition_val)

# create train and val dict
for i, img in enumerate(partition_train):
    partition_dict['train'].append(img)

for i, img in enumerate(partition_val):
    partition_dict['val'].append(img)
# =====


# print("partition_dict keys", partition_dict.keys())
print("Dataset sizes for train and val ============================================================")
print("partition list train and val len", len(partition_train), len(partition_val))
print("example from partition_train and partition_val list", partition_train[1], partition_val[1])
print("partition dict train and val len", len(partition_dict['train']), len(partition_dict['val']))


# # Generators
training_generator = DataGenerator(**params).generate(partition_dict['train'], n_stages, shuffle=True, augment=True, mode="train")


# # train_di = train_client.generate()  # original
# train_di = training_generator  # eggnog
# print("train_di", type(train_di),)
# train_samples =  len(partition_dict['train'])  # 100  # 117576  len(partition_dict['train'])


