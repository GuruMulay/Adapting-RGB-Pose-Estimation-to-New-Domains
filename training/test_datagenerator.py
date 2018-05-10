# test on /s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/s07_1video
# transform only works on one layout folder at a time

import sys
import os
sys.path.append("..")

from dataset_gen import DataGenerator  #g

split_videowise = True  #

n_stages = 1


if split_videowise:
    train_val_sessions = ['s07_1video']
    div_factor_train_val = 5
    print("train_val_sessions", train_val_sessions)
    print("div_factor_train_val", div_factor_train_val)
    
    
eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/"  # small dataset
batch_size = 2

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
          'save_transformed_path': None  # eggnog_dataset_path + train_val_sessions[0] + '_transformed/'
         }


partition_train = []
partition_val = []
partition_dict = {}
partition_dict['train'] = []
partition_dict['val'] = []


for session_name in train_val_sessions:
        for layout in [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]:
            for video_folder in [vf for vf in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if os.path.isdir(os.path.join(eggnog_dataset_path, session_name, layout, vf))]:
                files_list_video_folder = sorted(os.listdir(os.path.join(eggnog_dataset_path, session_name, layout, video_folder)))
                print("folder name === ", os.path.join(eggnog_dataset_path, session_name + '_transformed', layout, video_folder))
                params['save_transformed_path'] = os.path.join(eggnog_dataset_path, session_name + '_transformed', layout, video_folder)
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




# # Generators
params['save_transformed_path'] = os.path.join(eggnog_dataset_path, session_name + '_transformed_train', layout, video_folder)
training_generator = DataGenerator(**params).generate_and_save(partition_dict['train'], n_stages, shuffle=True, augment=True)


params['save_transformed_path'] = os.path.join(eggnog_dataset_path, session_name + '_transformed_val', layout, video_folder)
validation_generator = DataGenerator(**params).generate_and_save(partition_dict['val'], n_stages, shuffle=False, augment=True)

print("Testing datasetgen is done!")