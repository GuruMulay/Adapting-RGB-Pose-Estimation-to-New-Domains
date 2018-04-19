import sys
import os
import numpy as np
sys.path.append("..")
from glob import glob

from model import get_testing_model_eggnog
# from train_pose_1stage import get_last_epoch_and_weights_file  don't do this, it starts training

# 
import cv2
import matplotlib
import pylab as plt


# testing eggnog with only n stages
n_stages = 1

# sessions
val_sessions = ['s04']
test_sessions = val_sessions
div_factor_test = 5

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/"  # small dataset

BASE_DIR = "/s/red/b/nobackup/data/eggnog_cpm/from_scratch/0328180100pm/training/"
WEIGHTS_SAVE = 'weights_egg.{epoch:04d}.h5'
WEIGHT_DIR = BASE_DIR + "weights_egg"
OUTPUT_SAVE_DIR = BASE_DIR.replace('training', 'testing')
os.makedirs(OUTPUT_SAVE_DIR, exist_ok=True)

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


model = get_testing_model_eggnog(stages=n_stages)

# load stored weights
last_epoch, wfile = get_last_epoch_and_weights_file()
print("last epoch, wfile", last_epoch, wfile)


# populate the model with weights
print("Loading %s..." % wfile)
model.load_weights(wfile)



partition_test = []
# create val list
for session_name in test_sessions:
    for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
        for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf))]:
            print("test video_folder =====================", os.path.join(session_name, layout, video_folder))

            for file in sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder))):
                if file.endswith('.jpg') and "240x320" in file:
                    # if int(file.split('_')[-4])%div_factor_test == 0:  # append only if vfr number divisible by div_factor
                    if int(file.split('_')[-4])%div_factor_test == 0 and int(file.split('_')[-4])%2 != 0:  # append only if vfr number divisible by div_factor and odd
                        # print(file)
                        partition_test.append(session_name + "/" + layout + "/" +  video_folder + "/" + file[:-12])  # append the path from base dir = eggnog_dataset_dir


print("partition list test len", len(partition_test))
print("example from partition_train and partition_val list", partition_test[1])


# actual testing
test_index = 1

test_image = eggnog_dataset_path + partition_test[test_index] + '_240x320.jpg'
print("testing image", test_image)

oriImg = cv2.imread(test_image)  # B,G,R order
plt.imshow(oriImg[:,:,[2,1,0]])


# imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
input_img = np.transpose(np.float32(oriImg[:,:,:,np.newaxis]), (3,0,1,2)) # required shape (1, width, height, channels) 
print("Input shape: " + str(input_img.shape))

output_blobs = model.predict(input_img)
print("output len shapes", len(output_blobs), output_blobs[0][0].shape, output_blobs[1][0].shape)

# save pafs and hms
output_np_name = test_image.split('/')[-1][:-4]
print("saving to file ", OUTPUT_SAVE_DIR + output_np_name)
np.save(OUTPUT_SAVE_DIR + output_np_name + "_test_paf30x40.npy", output_blobs[0][0])
np.save(OUTPUT_SAVE_DIR + output_np_name + "_test_heatmap30x40.npy", output_blobs[1][0])
    
    
    
print("Testing done!")