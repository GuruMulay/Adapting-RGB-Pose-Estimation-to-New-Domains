import sys
import os
import pandas
import re
import math
sys.path.append("..")

from model import get_training_model_common
from dataset_gen import DataGenerator  # for eggnog
from ds_generators import DataIterator, DataGenCommon # for coco and for common DataGenCommon
from optimizers import MultiSGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras.layers.convolutional import Conv2D
from keras.applications.vgg19 import VGG19
import keras.backend as K

from glob import glob

from keras import callbacks
import random

from keras.backend import shape, int_shape
import pprint

from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig

"""
NOTE:
OFFLINE version
With this version:
Data can be read and augmented on-the-fly. 
Data can be read from *_augmented directories.

More:
This version trains on both COCO and EGGNOG simultaneously and val sets are 2: one for COCO and EGGNOG each.

"""

# for common set of joints between eggnog and coco
remove_joints = [0, 1, 2, 7, 11, 15, 16, 17, 18]  # total 9, so 19 - 9 = 10 common joints
# for removing 6 joints on two hands
# remove_joints = [7, 11, 15, 16, 17, 18]

def update_config_as_per_removed_joints():
    # update the config class instance
    rm_pairs = []
    rm_paf_xy = []
    
    for j in remove_joints:
        EggnogGlobalConfig.joint_indices.remove(j)

    EggnogGlobalConfig.n_hm = len(EggnogGlobalConfig.joint_indices)
    EggnogGlobalConfig.n_kp = len(EggnogGlobalConfig.joint_indices) - 1

    for p, pair in enumerate(EggnogGlobalConfig.paf_pairs_indices):  # 18
        print("p, pair", p, pair)
        if pair[0] in remove_joints or pair[1] in remove_joints:
            rm_pairs.append(pair)
            rm_paf_xy.append(2*p)  # add x paf map index to remove list
            rm_paf_xy.append(2*p+1)  # add y paf map index to remove list
            # following this way becasue you cannot do ".remove" on a list while enumerating over it.
    
    print("rm_pairs, rm_paf_xy", rm_pairs, rm_paf_xy)
    for rm in rm_pairs:
        EggnogGlobalConfig.paf_pairs_indices.remove(rm)
    for rm in rm_paf_xy:
        EggnogGlobalConfig.paf_indices_xy.remove(rm)

    
    EggnogGlobalConfig.n_paf = len(EggnogGlobalConfig.paf_pairs_indices)*2
    assert(len(EggnogGlobalConfig.paf_indices_xy) == EggnogGlobalConfig.n_paf)
    
    print("Updated joint info:")
    print("Final EggnogGlobalConfig.joint_indices", EggnogGlobalConfig.joint_indices)
    print("Final EggnogGlobalConfig.paf_pairs_indices", EggnogGlobalConfig.paf_pairs_indices)
    print("Final EggnogGlobalConfig.paf_indices", EggnogGlobalConfig.paf_indices_xy)
    print("EggnogGlobalConfig.n_kp, n_hm, n_paf", EggnogGlobalConfig.n_kp, EggnogGlobalConfig.n_hm, EggnogGlobalConfig.n_paf)

    
#g
verbose_print = True

update_config_as_per_removed_joints()

n_stages = 2
train_in_finetune_mode = False
preload_vgg = True
split_sessionwise = True  # e.g., s04 for training s07 for validation; OR split train and val sessionwise, 70% session for train and 30% session for val
branch_flag = 2  # 0 => both branches; 1 => branch L1 only; 2 => branch L2 only (heatmaps only) 
use_eggnong_common_joints = True 

if branch_flag == 0 or branch_flag == 1:
    raise NotImplementedError("L1 containing version is not written yet because we do not have 3 sets of pafs stored in pre-generated .npy files in the gt _augmented folders. Those pafs are neck to hipL; neck to hipR; and nose to neck.")

# stores train and val data
partition_dict = {}

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


# sessionwise split
if split_sessionwise:
    train_sessions = ['s01']  #, 's02']  #, 's03', 's08', 's09', 's10']
    val_sessions = ['s06']  # , 's07']
    
    # only take 1/div_factor fraction of data
    div_factor_train = 7
    div_factor_val = 70
    div_factor_aug = 3  # there are 5 versions of every frame after augmentation (_0, 1, 2, 3, 4.jpg) 
    
    print("train_sessions", train_sessions)
    print("val_sessions", val_sessions)
    print("div_factor_train", div_factor_train)
    print("div_factor_val", div_factor_val)
    print("div_factor_aug", div_factor_aug)


print("------------------ Flags ----------------------------")
print("train_in_finetune_mode", train_in_finetune_mode)
print("preload_vgg", preload_vgg)
print("split_sessionwise", split_sessionwise)
print("n_stages", n_stages)
print("branch_flag: [1 => branch L1 only; 2 => branch L2 only (heatmaps only)] ======", branch_flag)
print("use_eggnong_common_joints", use_eggnong_common_joints)
print("------------------ Flags ----------------------------")


batch_size = 5
base_lr = 1e-5
momentum = 0.9
weight_decay = 2e-5
lr_policy = "step"
gamma = 0.9  # originally 0.333
stepsize = 10000*17  # in original code each epoch is 121746 and step change is on 17th epoch
max_iter = 200
use_multiple_gpus = None  # set None for 1 gpu, not 1


os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

BASE_DIR = "/s/red/b/nobackup/data/eggnog_cpm/training_files/common_train/0614180100pm/training/"
print("creating a directory", BASE_DIR)
os.makedirs(BASE_DIR, exist_ok=True)
WEIGHTS_SAVE = 'weights_egg.{epoch:04d}.h5'
TRAINING_LOG = BASE_DIR + "training_eggnog.csv"
LOGS_DIR = BASE_DIR + "logs_egg"
WEIGHT_DIR = BASE_DIR + "weights_egg"


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


model = get_training_model_common(weight_decay, gpus=use_multiple_gpus, stages=n_stages, branch_flag=branch_flag)

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
    print("Fine tune mode (warm starting weights) ============================= ")
    cpm_weights_path = "../model/keras/model.h5"  # original weights converted from caffe
    model.load_weights(cpm_weights_path, by_name=True)  # only load everything except the final C blocks at every stage
    print("Loading the original CPM weights...", cpm_weights_path)
    last_epoch = 0
    
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
          'paf_n_channels': EggnogGlobalConfig.n_paf,
          'hm_height': 30,
          'hm_width': 40,
          'hm_n_channels': EggnogGlobalConfig.n_hm,
          'branch_flag': branch_flag,
          'save_transformed_path': None
         }
# '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/transformed/r2/'
# '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/transformed/r3/'


def prepare_train_val_data_dict_offline_version():
    # new version for eggnog dataset where images are distributed in sessions
    partition_train = []
    partition_val = []
    
    partition_dict['train'] = []
    partition_dict['val'] = []

    if split_sessionwise:   
        # go through sessions and add img path to lists
        # create train list
        for session_name in train_sessions:
            for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
                for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf)) and "augmented" in os.path.join(eggnog_dataset_path, session_name, layout, vf)]:
                    print("train video_folder =====================", os.path.join(session_name, layout, video_folder))

                    for file in sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder))):
                        if file.endswith('.jpg') and "240x320" in file:
                            if int(file.split('_')[-6])%div_factor_train == 0 and int(file.split('_')[-2])%div_factor_aug == 0 :  # append only if vfr number divisible by div_factor and aug_index is divisible by div_factor_aug
                                # print(file)
                                partition_train.append(session_name + "/" + layout + "/" +  video_folder + "/" + file.split("_240x320")[0])  # append the path from base dir = eggnog_dataset_dir

        # create val list
        for session_name in val_sessions:
            for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
                for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf)) and "augmented" in os.path.join(eggnog_dataset_path, session_name, layout, vf)]:
                    print("val video_folder =====================", os.path.join(session_name, layout, video_folder))

                    for file in sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder))):
                        if file.endswith('.jpg') and "240x320" in file:
                            if int(file.split('_')[-6])%div_factor_val == 0 and int(file.split('_')[-2])%div_factor_aug == 0:  # append only if vfr number divisible by div_factor and aug_index is divisible by div_factor_aug
                                # print(file)
                                partition_val.append(session_name + "/" + layout + "/" +  video_folder + "/" + file.split("_240x320")[0])  # append the path from base dir = eggnog_dataset_dir



    # shuffle train and val list
    random.seed(115)
    random.shuffle(partition_train)
    random.shuffle(partition_val)

    # create train and val dict
    for i, img in enumerate(partition_train):
        partition_dict['train'].append(img)

    for i, img in enumerate(partition_val):
        partition_dict['val'].append(img)


    # print("partition_dict keys", partition_dict.keys())
    print("Dataset sizes for train and val ============================================================")
    print("partition list train and val len", len(partition_train), len(partition_val))
    print("example from partition_train and partition_val list", partition_train[1], partition_val[1])
    print("partition dict train and val len", len(partition_dict['train']), len(partition_dict['val']))


prepare_train_val_data_dict_offline_version()


## Generators ##############################################

### EGGNOG
training_generator_eggnog = DataGenerator(**params)
validation_generator_eggnog = DataGenerator(**params)

train_di_eggnog = training_generator_eggnog.generate_with_masks(partition_dict['train'], n_stages, shuffle=True, augment=True, mode="train", online_aug=False, masking=True)  # eggnog
val_di_eggnog = validation_generator_eggnog.generate_with_masks(partition_dict['val'], n_stages, shuffle=False, augment=False, mode="val", online_aug=False, masking=True)  # eggnog

train_samples_eggnog = len(partition_dict['train'])  # 100  # 117576  len(partition_dict['train'])
val_samples_eggnog = len(partition_dict['val'])  # 30  # 2476  len(partition_dict['val'])
print("#### train_samples_eggnog, val_samples_eggnog", train_samples_eggnog, val_samples_eggnog)
# For eggnog full/5 => partition dict train and val len 88334 29879

### COCO
train_client_coco = DataIterator("/s/red/b/nobackup/data/eggnog_cpm/coco2014/train_dataset_2014.h5", shuffle=True, augment=True, batch_size=batch_size)
val_client_coco = DataIterator("/s/red/b/nobackup/data/eggnog_cpm/coco2014/val_dataset_2014.h5", shuffle=False, augment=False, batch_size=batch_size)

train_di_coco = train_client_coco.gen(n_stages, use_eggnong_common_joints, branch_flag=branch_flag)
val_di_coco = val_client_coco.gen(n_stages, use_eggnong_common_joints, branch_flag=branch_flag)

train_samples_coco = 10000  # 117576  # 100  # 
val_samples_coco = 1000  # 2476  # 30  # 
print("#### train_samples_coco, val_samples_coco", train_samples_coco, val_samples_coco)

## combined
# ##1 test with only coco
# train_di_eggnog = None
# val_di_eggnog = None
# train_samples_eggnog = 0
# val_samples_eggnog = 0
# ##

# ##2 test with only eggnog
# train_di_coco = None
# val_di_coco = None
# train_samples_coco = 0
# val_samples_coco = 0
# ##

train_gen_common = DataGenCommon(train_di_eggnog, train_di_coco)
val_gen_common = DataGenCommon(val_di_eggnog, val_di_coco)

train_di = train_gen_common.gen_common()
val_di = val_gen_common.gen_common()

train_samples = train_samples_eggnog + train_samples_coco
val_samples = val_samples_eggnog + val_samples_coco

##### Generators ##############################################


# learning rate schedule - equivalent of caffe lr_policy = "step"
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

callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan]

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

print("Training Completed!")