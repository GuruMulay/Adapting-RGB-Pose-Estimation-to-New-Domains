import sys
import os
import pandas
import re
import math
sys.path.append("..")

from model import get_training_model_eggnog
# from ds_generators import DataGeneratorClient, DataIterator
from dataset_gen import DataGenerator  #g
from optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19
import keras.backend as K

from glob import glob


#g
train_in_finetune_mode = False
preload_vgg = True


from keras import callbacks
import random

from keras.backend import shape, int_shape
import pprint

verbose_print = True

split_videowise = True  # split train and val for the same video, 70% frames for train and 30% frames for val
split_sessionwise = False  # split train and val sessionwise, 70% session for train and 30% session for val


# eggnog sessions split
# 1 staged 03201805pm lr = 2e-5
# 1 staged 0320180530pm higher lr (2e-4). (lr 2e-3 causes nan errors!)
# train_sessions = ['s20', 's08', 's09', 's10']
# val_sessions = ['s15']

# train_sessions = ['s04', 's05', 's06', 's16', 's20', 's08', 's09', 's10', 's11', 's12', 's17', 's21']
# val_sessions = ['s07', 's14', 's15']

# # 1 staged 0320180515pm
# train_sessions = ['s04', 's05', 's06', 's16', 's20', 's08', 's09', 's10', 's11', 's07', 's14', 's15']
# val_sessions = ['s12', 's17', 's21']

# train_sessions = ['s04', 's05']
# val_sessions = ['s05']


# test_sessions = ['s18', 's19']

# videowise split
if split_videowise:
    train_val_sessions = ['s04']
#     train_val_sessions = ['s04', 's05', 's07']
    div_factor_train_val = 5
    print("train_val_sessions", train_val_sessions)
    print("div_factor_train_val", div_factor_train_val)
    
    
# sessionwise split
if split_sessionwise:
    train_sessions = ['s04']
    val_sessions = ['s07']
    
#     # toy dataset2
#     train_sessions = ['s04_layout_p08_7v']
#     val_sessions = ['s04_layout_p08_3v']
     
    # only take 1/div_factor fraction of data
    div_factor_train = 5
    div_factor_val = 5
    
    print("train_sessions", train_sessions)
    print("val_sessions", val_sessions)
    print("div_factor_train", div_factor_train)
    print("div_factor_val", div_factor_val)
    


# testing eggnog with only n stages
n_stages = 1
print("------------------ Flags ----------------------------")
print("preload_vgg", preload_vgg)
print("split_videonwise", split_videowise)
print("split_sessionwise", split_sessionwise)
print("", )

"""
# notes
with 12:3:2 session, we have:
partition dict train and val len 88334 29879

"""

batch_size = 5
base_lr = 1e-5
momentum = 0.9
weight_decay = 2e-5
lr_policy = "step"
gamma = 0.333
stepsize = 10000*17 # in original code each epoch is 121746 and step change is on 17th epoch
max_iter = 200
use_multiple_gpus = None  # set None for 1 gpu, not 1

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

BASE_DIR = "/s/red/b/nobackup/data/eggnog_cpm/training_files/eggnog_preprocessing/0522180300pm/training/"
os.makedirs(BASE_DIR, exist_ok=True)
WEIGHTS_SAVE = 'weights_egg.{epoch:04d}.h5'
TRAINING_LOG = BASE_DIR + "training_eggnog.csv"
LOGS_DIR = BASE_DIR + "logs_egg"
WEIGHT_DIR = BASE_DIR + "weights_egg"


#g
class printbatch(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        print("logs on epoch begin ===========================", logs)
    def on_epoch_end(self, epoch, logs={}):
        print("logs on epoch end ===========================", logs)


def get_last_epoch_and_weights_file():
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    # os.makedirs(WEIGHT_DIR)
    files = [file for file in glob(WEIGHT_DIR + '/weights_egg.*.h5')]
    files = [file.split('/')[-1] for file in files]
    epochs = [file.split('.')[1] for file in files if file]
    epochs = [int(epoch) for epoch in epochs if epoch.isdigit() ]
    if len(epochs) == 0:
        if 'weights_egg.best.h5' in files:
            return -1, WEIGHT_DIR + '/weights_egg.best.h5'
    else:
        ep = max([int(epoch) for epoch in epochs])
        return ep, WEIGHT_DIR + '/' + WEIGHTS_SAVE.format(epoch=ep)
    return None, None


model = get_training_model_eggnog(weight_decay, gpus=use_multiple_gpus, stages=n_stages)


# if verbose_print:
#     print("model summary ====================================================== ", model.summary())
#     print("model config", model.get_config())
    
from_vgg = dict()
from_vgg['conv1_1'] = 'block1_conv1'
from_vgg['conv1_2'] = 'block1_conv2'
from_vgg['conv2_1'] = 'block2_conv1'
from_vgg['conv2_2'] = 'block2_conv2'
from_vgg['conv3_1'] = 'block3_conv1'
from_vgg['conv3_2'] = 'block3_conv2'
from_vgg['conv3_3'] = 'block3_conv3'
from_vgg['conv3_4'] = 'block3_conv4'
from_vgg['conv4_1'] = 'block4_conv1'
from_vgg['conv4_2'] = 'block4_conv2'

# load previous weights or vgg19 if this is the first run
last_epoch, wfile = get_last_epoch_and_weights_file()


if wfile is not None:
    print("Loading %s ..." % wfile)
    model.load_weights(wfile)
    last_epoch = last_epoch + 1

elif train_in_finetune_mode:  # load both cpm and vgg19 weights
    print("Loading the original CPM weights...")
    cpm_weights_path = "../model/keras/weights.0021.h5"  # original weights converted from caffe
    model.load_weights(cpm_weights_path, by_name=True)  # only load everything except the final C blocks at every stage

elif not preload_vgg:
    print("NOT loading vgg19 weights...")
    last_epoch = 0
    
else:  # only load vgg19 weights
    print("Loading vgg19 weights...")
    vgg_model = VGG19(include_top=False, weights='imagenet')

    for layer in model.layers:
        if layer.name in from_vgg:
            vgg_layer_name = from_vgg[layer.name]  # the 'block1_conv1' name from original VGG nomenclature
            layer.set_weights(vgg_model.get_layer(vgg_layer_name).get_weights())
            print("Loaded VGG19 layer: " + vgg_layer_name)

    last_epoch = 0

# setup lr multipliers for conv layers
lr_mult=dict()
for layer in model.layers:

    if isinstance(layer, Conv2D):

        # stage = 1
        if re.match("Mconv\d_stage1.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 1
            lr_mult[bias_name] = 2

        # stage > 1
        elif re.match("Mconv\d_stage.*", layer.name):
            kernel_name = layer.weights[0].name
            bias_name = layer.weights[1].name
            lr_mult[kernel_name] = 4
            lr_mult[bias_name] = 8

        # vgg
        else:
           kernel_name = layer.weights[0].name
           bias_name = layer.weights[1].name
           lr_mult[kernel_name] = 1
           lr_mult[bias_name] = 2

# configure loss functions

# euclidean loss as implemented in caffe https://github.com/BVLC/caffe/blob/master/src/caffe/layers/euclidean_loss_layer.cpp
def eucl_loss(x, y):
    l = K.sum(K.square(x - y)) / batch_size / 2
    return l

# prepare generators
# train_client = DataIterator("/s/red/b/nobackup/data/coco2014/train_dataset_2014.h5", shuffle=True, augment=True, batch_size=batch_size)
# val_client = DataIterator("/s/red/b/nobackup/data/coco2014/val_dataset_2014.h5", shuffle=False, augment=False, batch_size=batch_size)

# EGGNOG DataGenerator ==================================================================
# Parameters
# eggnog_dataset_path = '/s/red/b/nobackup/data/eggnog_cpm/test1_paf_hm/20160205_191417_00_Video'  # toy dataset

# eggonog sessions
eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/"  # original size dataset
# eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/"  # small dataset
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
# '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/transformed/r2/'


# old version for toy dataset where all the images were in the same folder
# # =====
# # create dictionaries with npy IDs
# partition_all = []
# partition_dict = {}
# partition_dict['train'] = []
# partition_dict['val'] = []

# for file in sorted(os.listdir(eggnog_dataset_path)):
#     if file.endswith('.jpg') and "240x320" not in file:
#         # print(file)
#         partition_all.append(file[:-4])

# print("partition_all before", partition_all[:10])
# random.seed(110)
# random.shuffle(partition_all)
# print("partition_all after", partition_all[:10])
# print("partition_all", len(partition_all))

# # divide all the data into 60:20:20 train:val:test
# for i, img in enumerate(partition_all):
#     if i < int(0.6*len(partition_all)):
#         partition_dict['train'].append(img)
#     elif int(0.6*len(partition_all)) <= i < int(0.8*len(partition_all)):
#         partition_dict['val'].append(img)
#     else:
#         doNothingForTestingSet = True
# # =====


# =====
# new version for eggnog dataset where images are distributed in sessions
partition_train = []
partition_val = []
partition_dict = {}
partition_dict['train'] = []
partition_dict['val'] = []

if split_sessionwise:   
    # go through sessions and add img path to lists
    # create train list
    for session_name in train_sessions:
        for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf))]:
                print("train video_folder =====================", os.path.join(session_name, layout, video_folder))

                for file in sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder))):
                    if file.endswith('.jpg') and "240x320" in file:
                        if int(file.split('_')[-4])%div_factor_train == 0:  # append only if vfr number divisible by div_factor
                            # print(file)
                            partition_train.append(session_name + "/" + layout + "/" +  video_folder + "/" + file[:-12])  # append the path from base dir = eggnog_dataset_dir

    # create val list
    for session_name in val_sessions:
        for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf))]:
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
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf))]:
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
random.seed(110)
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

# # Dataset ==================
# partition = {'train': ['20160205_191417_00_Video_vfr_95_skfr_94', '20160205_191417_00_Video_vfr_43_skfr_42', ...],
#               'val',: ['20160205_191417_00_Video_vfr_436_skfr_431', '20160205_191417_00_Video_vfr_49_skfr_49', ...]
#             }
# labels = # this dictionanry is same as the above one because for every val in partition_dict, you can get corresponding label 
# as key + '_heatmap240.npy' and key + '_paf240.npy'

# # Generators
training_generator = DataGenerator(**params).generate(partition_dict['train'], n_stages, shuffle=True, augment=True)
validation_generator = DataGenerator(**params).generate(partition_dict['val'], n_stages, shuffle=False, augment=False)


# train_di = train_client.generate()  # original
train_di = training_generator  # eggnog
print("train_di", type(train_di),)
train_samples =  len(partition_dict['train'])  # 100  # 117576  len(partition_dict['train'])
# val_di = val_client.generate()  # original
val_di = validation_generator  # eggnog
val_samples = len(partition_dict['val'])  # 30  # 2476  len(partition_dict['val'])
print("val_di", type(val_di),)
# For eggnog full/5 => partition dict train and val len 88334 29879


# ==================================================================

# original cpm with COCO ===========================================
# train_client = DataGenerator("/s/red/b/nobackup/data/coco2014/train_dataset_2014.h5", shuffle=True, augment=True, batch_size=batch_size)
# val_client = DataGenerator("/s/red/b/nobackup/data/coco2014/val_dataset_2014.h5", shuffle=False, augment=False, batch_size=batch_size)

# train_di = train_client.gen()
# print("train_di", type(train_di),)
# train_samples = 100  # 117576
# val_di = val_client.gen()
# val_samples = 30  # 2476
# print("val_di", type(val_di),)

# ===========================================


# learning rate schedule - equivalent of caffe lr_policy =  "step"
iterations_per_epoch = train_samples // batch_size

def step_decay(epoch):
    steps = epoch * iterations_per_epoch * batch_size
    lrate = base_lr * math.pow(gamma, math.floor(steps/stepsize))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate

print("Weight decay policy...")
for i in range(1,100,5): step_decay(i)

# configure callbacks
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHT_DIR + '/' + WEIGHTS_SAVE, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)
tnan = TerminateOnNaN()
pb = printbatch()

callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan, pb]

# sgd optimizer with lr multipliers
multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)


# start training
if use_multiple_gpus is not None:
    from keras.utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=use_multiple_gpus)

model.compile(loss=eucl_loss, optimizer=multisgd)

# if verbose_print:
#     for lyr in model.layers:
#         print(lyr.name, "==========> \n", "output shape:", lyr.output_shape)
# #         print("shape:", int_shape(lyr.output), shape(lyr.output))
# #         print("output shape:", lyr.output_shape)
# #         print("output shape:", lyr.get_output_at(0).get_shape().as_list())
# #         print("model config", model.get_config())


model.fit_generator(train_di,
                    steps_per_epoch=iterations_per_epoch,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    validation_data=val_di,
                    validation_steps=val_samples // batch_size,
                    use_multiprocessing=False,
                    initial_epoch=last_epoch
                    )

# print("Training Completed!")