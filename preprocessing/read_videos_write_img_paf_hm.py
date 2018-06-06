import sys
sys.path.append("..")

import os
import numpy as np
import cv2
import av
import PIL

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import skimage.io
from skimage.transform import resize, pyramid_reduce

# to merge preprocessing repo to keras repo
from py_eggnog_server.py_eggnog_heatmapper import Heatmapper
from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig, TransformationParams

# for augmentation
from py_eggnog_server.py_eggnog_transformer import Transformer, AugmentSelection


np.set_printoptions(threshold=np.nan)


# global variables
# eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/"  # for testing this python file
eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/"
# eggnog_dataset_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_s123/"  # for s01, s02, s03
n_aug_per_image = 5

### ********** ###
process_s010203 = False


# # heatmapper instantiated
# heatmapper = Heatmapper()


"""
# note 1:

The .frames file has same number of frames as the actual *Video.avi file.
1. For every frame in video, grab the corresponding timestamp from .frames file.
2. Go to rgb_skeleton file and find a timestamp closest to the timestamp from step 1
3. Return the closest timestamp from step 2 and correpsonding index (index in rgb_skeleton file)

# note 2:
For data splitting into 1/4th dataset: use vfr_number_ to check number%4==0, becuse vfr_number dows not repeat, while skfr_number repeats.

# note 3:
flag process_s010203 = True
For s01, s02, s03 we use *RGB.avi and *RGB.frames files instead of *Video.avi and *Video.frames 
Issue 1 (not really):
In directory /s/red/a/nobackup/cwc/data/userstudies/initial_study_xef/
From session s04 onwards, in *_layout_* folders (signalers) inside every session, we have *_Video.frames for every correpsonding  *_Video.avi file. But for sessions s01, s02, and s03 we DO NOT have *_Video.frames for every correpsonding  *_Video.avi file (maybe because these *_Video.avi are corrupted and in orders of Kbytes). However, we DO have *_RGB.frames for every correpsonding *_RGB.avi file (which makes s01, s02, and s03 usable). 

"""

def load_skeleton_data_for_video(video_file_path):
    """
    load the xyz and rgb skeleton files for given video 
    """
    xyz_skeleton_data = np.loadtxt(video_file_path[:-9] + 'Skeleton.txt', dtype='float', delimiter=',', skiprows=1, 
                                   usecols=(1,  # Time
                                            9,10,11,  # SpineBase 0
                                            18,19,20,  # SpineMid 
                                            27,28,29,  # Neck
                                            36,37,38,  # Head
                                            45,46,47,  # ShoulderLeft
                                            54,55,56,  # ElbowLeft
                                            63,64,65,  # WristLeft
                                            72,73,74,  # HandLeft
                                            81,82,83,  # ShoulderRight
                                            90,91,92,  # ElbowRight
                                            99,100,101,  # WristRight
                                            108,109,110,  # HandRight
                                            117,118,119,  # HipLeft (added)
                                            # 126,127,128,  # KneeLeft
                                            # 135,136,137,  # AnkleLeft
                                            # 144,145,146,  # FootLeft
                                            153,154,155,  # HipRight (added)
                                            # 162,163,164,  # KneeRight
                                            # 171,172,173,  # AnkleRight
                                            # 180,181,182,  # FootRight
                                            189,190,191,  # SpineShoulder
                                            198,199,200,  # HandTipLeft
                                            207,208,209,  # ThumbLeft
                                            216,217,218,  # HandTipRight
                                            225,226,227  # ThumbRight 
                                           ))
    
    assert(xyz_skeleton_data.shape[1] == 58) # 19 joints * 3 (x,y,z) + 1 (time)
    print("xyz_skeleton_data.shape", xyz_skeleton_data.shape)
    

    visible_skeleton_kp = [7,8,9,  # SpineBase 0
                           15,16,17,  # SpineMid
                           23,24,25,  # Neck
                           31,32,33,  # Head
                           39,40,41,  # ShoulderLeft
                           47,48,49,  # ElbowLeft
                           55,56,57,  # WristLeft
                           63,64,65,  # HandLeft
                           71,72,73,  # ShoulderRight
                           79,80,81,  # ElbowRight
                           87,88,89,  # WristRight
                           95,96,97,  # HandRight
                           103,104,105,  # HipLeft
                           # 111,112,113,  # KneeLeft
                           # 119,120,121,  # AnkleLeft
                           # 127,128,129,  # FootLeft
                           135,136,137,  # HipRight
                           # 143,144,145,  # KneeRight
                           # 151,152,153,  # AnkleRight
                           # 159,160,161,  # FootRight
                           167,168,169,  # SpineShoulder
                           175,176,177,  # HandTipLeft
                           183,184,185,  # ThumbLeft
                           191,192,193,  # HandTipRight
                           199,200,201  # ThumbRight
                          ]
        
        
    symbol_inf_neg = "-∞"
    symbol_inf_pos = "∞"
    rgb_skeleton_data = []
    
    with open(video_file_path[:-9] + 'RGBSkeleton.txt') as rgb_sk:
        next(rgb_sk)
        line_index = 0
        for line in rgb_sk:
            cols = line.split(',')
            cols_actual_data = [cols[i] for i in visible_skeleton_kp]
            if line_index < 0:
                print("before replacement ===== ", len(cols), len(cols_actual_data), cols_actual_data)
            
            for j, field in enumerate(cols_actual_data):
                if symbol_inf_neg in field or symbol_inf_pos in field:
                    cols_actual_data[j] = np.inf  # to replace "∞" symbols with np.inf
                elif "UNTRACKED" in field:
                    cols_actual_data[j] = 0
                elif "INFERRED" in field:
                    cols_actual_data[j] = 1
                elif "TRACKED" in field:
                    cols_actual_data[j] = 2
                else:
                    cols_actual_data[j] = np.float64(field)
            # append 'time' from column 0 of the xyz_skeleton_data to the column 0 of cols_actual_data
            cols_actual_data.insert(0, xyz_skeleton_data[line_index][0])
            if line_index < 0:
                print("after replacement ===== ", len(cols), len(cols_actual_data), cols_actual_data)
            
            line_index += 1
            assert(len(cols_actual_data) == 58)  # 19 joints * 3 (tracking_info,x,y) + 1 (Time)
            rgb_skeleton_data.append(cols_actual_data)
            
            
    rgb_skeleton_data = np.array(rgb_skeleton_data)
    print("rgb_skeleton_data.shape", rgb_skeleton_data.shape)
    
    # assert that there are same number of lines on both skeleton files
    assert(xyz_skeleton_data.shape[0] == rgb_skeleton_data.shape[0])
    assert((xyz_skeleton_data[...,0] == rgb_skeleton_data[...,0]).all())
    
    return xyz_skeleton_data, rgb_skeleton_data


def load_skeleton_data_for_video_v1(video_file_path):
    """
    load the xyz and rgb skeleton files for given video 
    """
    xyz_skeleton_data = np.loadtxt(video_file_path[:-9] + 'Skeleton.txt', dtype='float', delimiter=',', skiprows=1, 
                                   usecols=(1,  # Time
                                            9,10,11,  # SpineBase 0
                                            18,19,20,  # SpineMid 1
                                            27,28,29,  # Neck 2
                                            36,37,38,  # Head 3
                                            45,46,47,  # ShoulderLeft 4
                                            54,55,56,  # ElbowLeft 5
                                            63,64,65,  # WristLeft 6
                                            72,73,74,  # HandLeft 7
                                            81,82,83,  # ShoulderRight 8
                                            90,91,92,  # ElbowRight 9
                                            99,100,101,  # WristRight 10
                                            108,109,110,  # HandRight 11
                                            117,118,119,  # HipLeft (added) 12
                                            # 126,127,128,  # KneeLeft
                                            # 135,136,137,  # AnkleLeft
                                            # 144,145,146,  # FootLeft
                                            153,154,155,  # HipRight (added) 13
                                            # 162,163,164,  # KneeRight
                                            # 171,172,173,  # AnkleRight
                                            # 180,181,182,  # FootRight
                                            189,190,191,  # SpineShoulder 14
                                            198,199,200,  # HandTipLeft 15
                                            207,208,209,  # ThumbLeft 16
                                            216,217,218,  # HandTipRight 17
                                            225,226,227  # ThumbRight 18
                                           ))
    
    assert(xyz_skeleton_data.shape[1] == 58) # 19 joints * 3 (x,y,z) + 1 (time)
    print("xyz_skeleton_data.shape", xyz_skeleton_data.shape)
    
#     rgb_skeleton_data = np.loadtxt(video_file_path[:-9] + 'RGBSkeleton.txt', dtype='float', delimiter=',', skiprows=1, 
#                                    converters=cnv, 
#                                    usecols=(8,9,  
#                                              16,17,
#                                              24,25,
#                                              32,33,
#                                              40,41,
#                                              48,49,
#                                              56,57,
#                                              64,65,
#                                              72,73,
#                                              80,81,
#                                              88,89,
#                                              96,97,
#                                              104,105,  # HipLeft
#                                              # 112,113,  # KneeLeft
#                                              # 120,121,  # AnkleLeft
#                                              # 128,129,  # FootLeft
#                                              136,137,
#                                              # 144,145,  # KneeRight
#                                              # 152,153,  # AnkleRight
#                                              # 160,161,  # FootRight
#                                              168,169,  # SpineShoulder
#                                              176,177,  # HandTipLeft
#                                              184,185,  # ThumbLeft
#                                              192,193,  # HandTipRight
#                                              200,201  # ThumbRight
#                                             ))

#     visible_skeleton_kp = [7,8,9,  # SpineBase 0
#                            15,16,17,  # SpineMid
#                            23,24,25,  # Neck
#                            31,32,33,  # Head
#                            39,40,41,  # ShoulderLeft
#                            47,48,49,  # ElbowLeft
#                            55,56,57,  # WristLeft
#                            63,64,65,  # HandLeft
#                            71,72,73,  # ShoulderRight
#                            79,80,81,  # ElbowRight
#                            87,88,89,  # WristRight
#                            95,96,97,  # HandRight
#                            103,104,105,  # HipLeft
#                            # 111,112,113,  # KneeLeft
#                            # 119,120,121,  # AnkleLeft
#                            # 127,128,129,  # FootLeft
#                            135,136,137,  # HipRight
#                            # 143,144,145,  # KneeRight
#                            # 151,152,153,  # AnkleRight
#                            # 159,160,161,  # FootRight
#                            167,168,169,  # SpineShoulder
#                            175,176,177,  # HandTipLeft
#                            183,184,185,  # ThumbLeft
#                            191,192,193,  # HandTipRight
#                            199,200,201  # ThumbRight
#                           ]
    
    visible_skeleton_kp = [8,9,  # SpineBase 0
                           16,17,  # SpineMid
                           24,25,  # Neck
                           32,33,  # Head
                           40,41,  # ShoulderLeft
                           48,49,  # ElbowLeft
                           56,57,  # WristLeft
                           64,65,  # HandLeft
                           72,73,  # ShoulderRight
                           80,81,  # ElbowRight
                           88,89,  # WristRight
                           96,97,  # HandRight
                           104,105,  # HipLeft
                           # 112,113,  # KneeLeft
                           # 120,121,  # AnkleLeft
                           # 128,129,  # FootLeft
                           136,137,  # HipRight
                           # 144,145,  # KneeRight
                           # 152,153,  # AnkleRight
                           # 160,161,  # FootRight
                           168,169,  # SpineShoulder
                           176,177,  # HandTipLeft
                           184,185,  # ThumbLeft
                           192,193,  # HandTipRight
                           200,201  # ThumbRight
                          ]
        
        
    symbol_inf_neg = "-∞"
    symbol_inf_pos = "∞"
    rgb_skeleton_data = []
    
    with open(video_file_path[:-9] + 'RGBSkeleton.txt') as rgb_sk:
        next(rgb_sk)
        line_index = 0
        for line in rgb_sk:
            cols = line.split(',')
            cols_actual_data = [cols[i] for i in visible_skeleton_kp]
            # print(len(cols), len(cols_actual_data), cols_actual_data)
            
            for j, field in enumerate(cols_actual_data):
                if symbol_inf_neg in field or symbol_inf_pos in field:
                    cols_actual_data[j] = np.inf  # to replace "∞" symbols with np.inf
                else:
                    cols_actual_data[j] = np.float64(field)
            # append 'time' from column 0 of the xyz_skeleton_data to the column 0 of cols_actual_data
            cols_actual_data.insert(0, xyz_skeleton_data[line_index][0])
            # print(len(cols), len(cols_actual_data), cols_actual_data)
            line_index += 1
            assert(len(cols_actual_data) == 39)  # 19 joints * 2 (x,y) + 1 (Time)
            rgb_skeleton_data.append(cols_actual_data)
            
            
    rgb_skeleton_data = np.array(rgb_skeleton_data)
    print("rgb_skeleton_data.shape", rgb_skeleton_data.shape)
    
    # assert that there are same number of lines on both skeleton files
    assert(xyz_skeleton_data.shape[0] == rgb_skeleton_data.shape[0])
    assert((xyz_skeleton_data[...,0] == rgb_skeleton_data[...,0]).all())
    
    return xyz_skeleton_data, rgb_skeleton_data


def load_frames_file(video_file_path):
    """
    loads the frames file for given video and returns a dictionary with {frame_no: frame_time} format
    """

    frame_time_dict = dict()
    with open(video_file_path[:-4] + '.frames') as frames:
        next(frames)
        for line in frames:
            cols = line.split(',')
            frame_time_dict[np.int64(cols[0])] = np.float64(cols[1])
            
    print("total frames in the video =", len(frame_time_dict))
    return frame_time_dict


def find_nearest_frameindex_from_skeleton_file(sk_time_array, time):
    idx = (np.abs(sk_time_array - time)).argmin()
    return idx, sk_time_array[idx]


# _v1 older version under preprocessing repo; newer version is under keras-*/preprocesssing folder

def get_heatmap(index_array, pxpy_list, tracking_state):
    # index_array (240/8, 320/8, 2), pxpy_list [px, py], tracking_state 0, 1, or 2
    # alpha = 1.5
    alpha = TransformationParams.alpha
    kp_location_array = np.zeros(index_array.shape)
    assert(index_array.shape[0] > 0 and index_array.shape[1] > 0)
    
    if pxpy_list[0] is None:
        heatmap = kp_location_array[:,:,0]
    else:
        if tracking_state == 0:  # meaning that the joint was not tracked in that frame, so return all black image
            print("untracked kp")
            heatmap = kp_location_array[:,:,0]
        else:
            kp_location_array[:, :, 0] = pxpy_list[1]
            kp_location_array[:, :, 1] = pxpy_list[0]
            heatmap = np.exp((alpha)*-np.sqrt(np.sum(np.square(index_array - kp_location_array), axis=2)))
            heatmap /= np.max(heatmap)
            
            if tracking_state == 1:  # joint is inferred, so return grey gaussian instead of white one?
                # print("inferred kp")
                heatmap /= 2  # using this the white peak of gaussian is reduced in its magnitude by a factor of 2
                
    return heatmap


def get_pafx_pafy(index_array, kp0xy, kp1xy, tracking_states_pair):
    """
    kp0xy, kp1xy: lists [pixel_x, pixel_y] for kp0 and kp1
    tracking_states_pair = [0, 0] or [1, 0] etc
    """
    # limb_width = 1.0  # pixels  # for low res pafs
    limb_width = TransformationParams.limb_width 
    # limb_width = 1.25*4  # pixels  # for high res pafs used for verification
    assert(index_array.shape[0] > 0 and index_array.shape[1] > 0)
    
    paf_array = np.zeros(index_array.shape)  # stores calculated pafs (values between -1 to 1)
    
    
    if 0 in tracking_states_pair:  # if one of the joints is UNTRACKED (0) then return zeroed paf_array
        print("tracking_states_pair", tracking_states_pair)
        return paf_array
    
    p_vector_array = np.zeros(index_array.shape)  # slice 0 stores all y, slice 1 stores all x
    
    # p_vector_array correpsonds to non-unit vector (p - x(j1, k)) from the paper # j1 is kp0
    # swapped indexing (0, 1) because slice 0 of index array stores all y locations and slice 1 stores all x locations
    p_vector_array[:,:,0] = index_array[:,:,0] - kp0xy[1]
    p_vector_array[:,:,1] = index_array[:,:,1] - kp0xy[0]
    
    # v_vector_array corresponds to v from the paper
    vect = np.array((kp1xy[1] - kp0xy[1], kp1xy[0] - kp0xy[0])).reshape(1, -1)  # print("vect", vect)
    v_unit_arr, limb_length_arr = normalize(vect, return_norm=True)  # y component at 0th index # x component at 1st index
    v_unit = v_unit_arr[0]
    limb_length = limb_length_arr[0]  # print("vect unit and limb length", v_unit, limb_length)
    v_perpendicular = [v_unit[1], -v_unit[0]]  # print("v_perp", v_perpendicular)
    #     v_vector_array[:,:,0] = v_unit[0]  # y component at 0th index
    #     v_vector_array[:,:,1] = v_unit[1]  # x component at 1st index
    
    # print("generating paf for the limb formed by", kp0xy, kp1xy)
    # paf_array = index_array  # this caused gray gradient error
    
    for r in range(paf_array.shape[0]):
        for c in range(paf_array.shape[1]):
            # print("r, c = ", r, c)
            # check if p is "on the limb"
            if (0 <= np.dot(v_unit, [p_vector_array[r,c,0], p_vector_array[r,c,1]]) <= limb_length) and (np.abs(np.dot(v_perpendicular, [p_vector_array[r,c,0], p_vector_array[r,c,1]])) <= limb_width):
                paf_array[r, c, 1] = v_unit[0] # y component of the expected vector is assigned to the 1st channel so that we append PAFs in x and then y order
                paf_array[r, c, 0] = v_unit[1] # x component of the expected vector is assigned to the 0th channel so that we append PAFs in x and then y order
#                 print("x, y", v_unit[1], v_unit[0])
#             else:
#                 print("should assign zero")
    # return paf so that x slice is first (index 0) and the y slice is second (index 1)
    
    return paf_array



def kpx_kpy_transformer(kp_list):
    """
    transform kps from 1920x1080 to 320x240 space and then to 40x30 ground truth space
    """
    # for normal res pafs
    kpx_transformed = (kp_list[0]-EggnogGlobalConfig.kp_x_offset_half) / EggnogGlobalConfig.kp_to_img_stride/EggnogGlobalConfig.ground_truth_factor
    kpy_transformed = (kp_list[1])/EggnogGlobalConfig.kp_to_img_stride/EggnogGlobalConfig.ground_truth_factor
    
    # for high res pafs, this was used for verification
    # kpx_transformed = (kp_list[0]-240)/4.5
    # kpy_transformed = (kp_list[1])/4.5
    
    return [kpx_transformed, kpy_transformed]


    
def save_2d_keypoints_and_images_v1(video_name, video_path, npy_path, rgb_skeleton_data, frame_time_dict):
    mismatch_count = 0
    ## cap = cv2.VideoCapture(video_path)
    ## assert(cap.isOpened() == True)
    
    container = av.open(video_path)
    
    for k, fr in enumerate(container.decode(video=0)):
        assert(k == fr.index)
    ## for k in frame_time_dict.keys():
        nearest_idx, nearest_time = find_nearest_frameindex_from_skeleton_file(rgb_skeleton_data[...,0], frame_time_dict[k]) # take column 0 (time) from rgb data
        # print("k (video frame) ", k, "\t time", frame_time_dict[k], "\t nearest_idx from skeleton file", nearest_idx, "\t nearest_time", nearest_time)  # print("k=>", k, nearest_idx, "<= nearest_idx")
       
        if (abs(frame_time_dict[k] - nearest_time) > 1000000):  # 100 ns ticks, so 1000000 = 0.1sec
            mismatch_count += 1
            continue  # do not add the nearest found index if the difference is really big (>0.1sec)
        else:         
            # print(rgb_skeleton_data[nearest_idx])
            if(np.inf not in rgb_skeleton_data[nearest_idx]):  # do not add if there is np.inf in the line
                
                ## cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                ## success, frame = cap.read()  # frame is read as (h, w, c)
                
                success = True  # hard-coded for PyAV
                frame = fr.to_image()
                # converting PIL (<class 'PIL.Image.Image'>) to <class 'numpy.ndarray'>
                img = np.asarray(frame)  # h, w, c
                
                if success:
                    os.makedirs(os.path.join(npy_path, video_name), exist_ok=True)
                    save_dir = os.path.join(npy_path, video_name)
                    
                    # 1
                    # save image with the original resolution
                    # print("kth frame =", k, frame.shape, "\n")
                    # cv2.imwrite(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + '.jpg'), frame)

                    # 2
                    # save downsampled image
                    ## bgr to rgb
                    ## img = frame[...,::-1]
                    img_central = img[:, EggnogGlobalConfig.kp_x_offset_half:(EggnogGlobalConfig.original_w-EggnogGlobalConfig.kp_x_offset_half), :]
                    # downsample by 4.5
                    img_down = pyramid_reduce(img_central, downscale=EggnogGlobalConfig.kp_to_img_stride)  # better than resize
                    # print("img_down shape (h, w, c)", img_down.shape)  # height, width, channels (rgb) 
                    skimage.io.imsave(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_240x320.jpg"), img_down)
                    
                    
                    # 3
                    # save heatmaps skimage.io.imsave(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_240x320.jpg"), img_down)and pafs
                    sk_keypoints_with_tracking_info = rgb_skeleton_data[nearest_idx][1:]  # ignore index 0 (time)
                    sk_kp_tracking_info = sk_keypoints_with_tracking_info[np.arange(0, sk_keypoints_with_tracking_info.size, 3)]  # only the info about tracking states (0, 1, 2) for 19 joints
                    sk_keypoints = np.delete(sk_keypoints_with_tracking_info, np.arange(0, sk_keypoints_with_tracking_info.size, 3))  # this is without tracking info, by removing the tracking info
                    # print("sk_kp shape =", sk_keypoints.shape)  # (38, )
                    
                    # for 20 (actually 19 + background) heatmaps =====================================
                    for kpn in range(sk_keypoints.shape[0]//2):
                        kpx = sk_keypoints[2*kpn]
                        kpy = sk_keypoints[2*kpn+1]  # print(kpx, kpy)
                        tracking_state = sk_kp_tracking_info[kpn]
                        
                        # index_array = np.zeros((240//ground_truth_factor, 320//ground_truth_factor, 2))
                        index_array = np.zeros((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, 2))
                        for i in range(index_array.shape[0]):
                            for j in range(index_array.shape[1]):
                                index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                
                        if kpn == 0:
                            heatmap = get_heatmap(index_array, kpx_kpy_transformer([kpx, kpy]), tracking_state)   # /4 because image is 1080 x 1920 and so are the original pixel locations of the keypoints 
                        else:
                            heatmap = np.dstack(( heatmap, get_heatmap(index_array, kpx_kpy_transformer([kpx, kpy]), tracking_state) ))
                        # print("heatmap.shape =", heatmap.shape)
            
                    # generate background heatmap
                    maxed_heatmap = np.max(heatmap[:,:,:], axis=2)  # print("maxed_heatmap.shape = ", maxed_heatmap.shape)
            
                    heatmap = np.dstack((heatmap, 1 - maxed_heatmap))
                    # print("final heatmap.shape =", heatmap.shape)
                    np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_heatmap30x40.npy"), heatmap)
            
            
                    # for 18x2 PAFs =====================================
                    for n, pair in enumerate(EggnogGlobalConfig.paf_pairs_indices):
                        # print("writing paf for index", n, pair)
                        index_array = np.zeros((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, 2))
                        for i in range(index_array.shape[0]):
                            for j in range(index_array.shape[1]):
                                index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                        
                        tracking_states = [sk_kp_tracking_info[pair[0]], sk_kp_tracking_info[pair[1]]]
                        
                        if n == 0:
                            paf = get_pafx_pafy(index_array, 
                                        kp0xy=kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                        kp1xy=kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]),
                                        tracking_states_pair=tracking_states
                                        )
                        else:
                            paf = np.dstack(( paf,  get_pafx_pafy(index_array, 
                                        kp0xy=kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                        kp1xy=kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]),
                                        tracking_states_pair=tracking_states
                                        )
                                    ))
                        # print("paf.shape =", paf.shape)

                    
                    # print("final paf.shape =========================", paf.shape)
                    np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_paf30x40.npy"), paf)
            
                    
                    # 4
                    # save the 2d keypoints of shape (38,)
                    # print(rgb_skeleton_data[nearest_idx])
                    # print(save_dir, os.path.join("", video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + '.npy'))
                    np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + '.npy'), rgb_skeleton_data[nearest_idx][1:])  # index 0 is time # saving all 57 values 19 * 3 (tracking, x, y)
    
    
    ## cap.release()
    print("mismatch_count =",  mismatch_count)
    

def transform_data(img, kp, kp_tracking_info, augment):
    """
    img: 240x320 RGB image
    kp: these are in 1920x1080 space (38, )  19*2 (x,y)`
    kp_tracking_info: tracking info for 19 joints (19, )
    augment Boolean
    """
    
    aug = AugmentSelection.random() if augment else AugmentSelection.unrandom()
#         print("transform data: before transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)  
        # transform data: before transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
        # transform data: after transform (240, 320, 3) (30, 40, 36) (30, 40, 20) (38,)
    ### aug.print_aug_params()
    
    # Transformer instantiated
    transformer = Transformer()
    
    ### print("kp info", kp, kp_tracking_info)
    img, kp, kp_tracking_info = transformer.transform(img, kp, kp_tracking_info, aug=aug)
#         print("transform data: after transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)

    # Heatmapper instantiated
    heatmapper = Heatmapper()
    label_paf, label_hm = heatmapper.get_pafs_and_hms_heatmaps(kp, kp_tracking_info)  # these kp are in the image space (240x320)
    
    
    return img, label_paf, label_hm, kp

    
    
def save_2d_keypoints_and_images_v2(video_name, video_path, npy_path, rgb_skeleton_data, frame_time_dict):
    mismatch_count = 0
    
    container = av.open(video_path)
    for k, fr in enumerate(container.decode(video=0)):
        # print("frame k ==========================", k)
        assert(k == fr.index)
        nearest_idx, nearest_time = find_nearest_frameindex_from_skeleton_file(rgb_skeleton_data[...,0], frame_time_dict[k]) # take column 0 (time) from rgb data
       
        if (abs(frame_time_dict[k] - nearest_time) > 1000000):  # 100 ns ticks, so 1000000 = 0.1sec
            mismatch_count += 1
            continue  # do not add the nearest found index if the difference is really big (>0.1sec)
        else:         
            # print(rgb_skeleton_data[nearest_idx])
            if(np.inf not in rgb_skeleton_data[nearest_idx]):  # do not add if there is np.inf in the line
                
                success = True  # hard-coded for PyAV
                frame = fr.to_image()
                # converting PIL (<class 'PIL.Image.Image'>) to <class 'numpy.ndarray'>
                img = np.asarray(frame)  # h, w, c
                
                if success:
                    os.makedirs(os.path.join(npy_path, video_name), exist_ok=True)
                    save_dir = os.path.join(npy_path, video_name)
                    
                    os.makedirs(os.path.join(npy_path, video_name + "_augmented"), exist_ok=True)
                    save_dir_augmented = os.path.join(npy_path, video_name + "_augmented")
                    
                    # 1
                    # save image with the original resolution
                    # cv2.imwrite(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + '.jpg'), frame)

                    # 2
                    # save downsampled image
                    img_central = img[:, EggnogGlobalConfig.kp_x_offset_half:(EggnogGlobalConfig.original_w-EggnogGlobalConfig.kp_x_offset_half), :]
                    # downsample by 4.5
                    img_down = pyramid_reduce(img_central, downscale=EggnogGlobalConfig.kp_to_img_stride)  # better than resize
                    skimage.io.imsave(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_240x320.jpg"), img_down)  # 240x320
                    
                    
                    # 3
                    # save heatmaps skimage.io.imsave(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_240x320.jpg"), img_down)and pafs
                    sk_keypoints_with_tracking_info = rgb_skeleton_data[nearest_idx][1:]  # ignore index 0 (time)
                    sk_kp_tracking_info = sk_keypoints_with_tracking_info[np.arange(0, sk_keypoints_with_tracking_info.size, 3)]  # only the info about tracking states (0, 1, 2) for 19 joints
                    sk_keypoints = np.delete(sk_keypoints_with_tracking_info, np.arange(0, sk_keypoints_with_tracking_info.size, 3))  # this is without tracking info, by removing the tracking info
                    # print("sk_kp shape =", sk_keypoints.shape)  # (38, )
                    
                    
                    ################################
                    # save augmented (transformed) images and their ground truths
                    for a in range(n_aug_per_image):
                        # print("aug index", a)
#                         X = np.empty((EggnogGlobalConfig.height, EggnogGlobalConfig.width, 3), dtype=np.uint8)
                        # print(type(X), X.dtype)  # uint8
#                         y1 = np.empty((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.n_paf))
#                         y2 = np.empty((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.n_hm))
#                         kp_transformed = np.empty((EggnogGlobalConfig.n_kp*2))  # (batch x (20-1)*2)  # e.g., 5x38
                        
                        # somehow feeding in img_down does not work! So this is a workaround
                        img_read = skimage.io.imread(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_240x320.jpg"))
                        sk_keypoints_with_tracking_info = rgb_skeleton_data[nearest_idx][1:]  # ignore index 0 (time)
                        sk_kp_tracking_info = sk_keypoints_with_tracking_info[np.arange(0, sk_keypoints_with_tracking_info.size, 3)]  # only the info about tracking states (0, 1, 2) for 19 joints
                        sk_keypoints = np.delete(sk_keypoints_with_tracking_info, np.arange(0, sk_keypoints_with_tracking_info.size, 3))  # this is without tracking info, by removing the tracking info
                        
                        # HAD to reread the sk_* becasue following transform data function modifies them for every call
                        ### print("kp info", sk_keypoints, sk_kp_tracking_info)
                        X, y1, y2, kp_transformed = transform_data(img_read,
                                        sk_keypoints,  # without tracking info
                                        sk_kp_tracking_info,  # tracking info only
                                        augment=True)
                        
                        # print(type(X), X.dtype, X.astype(np.uint8).dtype) # <class 'numpy.ndarray'> float64 uint8 
                        # use astype to convert from float64 to uint8 
                        skimage.io.imsave(os.path.join(save_dir_augmented, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_aug_" + str(a) + "_240x320.jpg"), X.astype(np.uint8))
                        
                        np.save(os.path.join(save_dir_augmented, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_aug_" + str(a) + "_paf30x40.npy"), y1)  # pafs
                        
                        np.save(os.path.join(save_dir_augmented, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_aug_" + str(a) + "_heatmap30x40.npy"), y2)  # heatmaps
                        
                    ################################
                    
                    # for 20 (actually 19 + background) heatmaps =====================================
                    for kpn in range(sk_keypoints.shape[0]//2):
                        kpx = sk_keypoints[2*kpn]
                        kpy = sk_keypoints[2*kpn+1]  # print(kpx, kpy)
                        tracking_state = sk_kp_tracking_info[kpn]
                        
                        # index_array = np.zeros((240//ground_truth_factor, 320//ground_truth_factor, 2))
                        index_array = np.zeros((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, 2))
                        for i in range(index_array.shape[0]):
                            for j in range(index_array.shape[1]):
                                index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                
                        if kpn == 0:
                            heatmap = get_heatmap(index_array, kpx_kpy_transformer([kpx, kpy]), tracking_state)   # /4 because image is 1080 x 1920 and so are the original pixel locations of the keypoints 
                        else:
                            heatmap = np.dstack(( heatmap, get_heatmap(index_array, kpx_kpy_transformer([kpx, kpy]), tracking_state) ))
                        # print("heatmap.shape =", heatmap.shape)
            
                    # generate background heatmap
                    maxed_heatmap = np.max(heatmap[:,:,:], axis=2)  # print("maxed_heatmap.shape = ", maxed_heatmap.shape)
            
                    heatmap = np.dstack((heatmap, 1 - maxed_heatmap))
                    # print("final heatmap.shape =", heatmap.shape)
                    np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_heatmap30x40.npy"), heatmap)
            
            
                    # for 18x2 PAFs =====================================
                    for n, pair in enumerate(EggnogGlobalConfig.paf_pairs_indices):
                        # print("writing paf for index", n, pair)
                        index_array = np.zeros((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, 2))
                        for i in range(index_array.shape[0]):
                            for j in range(index_array.shape[1]):
                                index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                        
                        tracking_states = [sk_kp_tracking_info[pair[0]], sk_kp_tracking_info[pair[1]]]
                        
                        if n == 0:
                            paf = get_pafx_pafy(index_array, 
                                        kp0xy=kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                        kp1xy=kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]),
                                        tracking_states_pair=tracking_states
                                        )
                        else:
                            paf = np.dstack(( paf,  get_pafx_pafy(index_array, 
                                        kp0xy=kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                        kp1xy=kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]),
                                        tracking_states_pair=tracking_states
                                        )
                                    ))
                        # print("paf.shape =", paf.shape)

                    
                    # print("final paf.shape =========================", paf.shape)
                    np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_paf30x40.npy"), paf)
            
                    
                    # 4
                    # save the 2d keypoints of shape (38,) in original 1920x1080 space
                    # print(rgb_skeleton_data[nearest_idx])
                    # print(save_dir, os.path.join("", video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + '.npy'))
                    np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + '.npy'), rgb_skeleton_data[nearest_idx][1:])  # index 0 is time # saving all 57 values 19 * 3 (tracking, x, y)
    
    
    print("mismatch_count =",  mismatch_count)
    
    
def process_session(session_name):  
    print("processing the session with path", eggnog_dataset_path + session_name)
    
    layout_sessions = [l for l in os.listdir(os.path.join(eggnog_dataset_path, session_name)) if "layout" in l]
    
    for layout in layout_sessions:
        print("processing layout =========================================", layout)
        videos = [v for v in os.listdir(os.path.join(eggnog_dataset_path, session_name, layout)) if ".avi" in v]
        for video in videos:
            print("processing video", video)
            
            video_file_path = os.path.join(eggnog_dataset_path, session_name, layout, video)
            # rename and replace 'RGB' with 'Video' in the video file and frames file (see #note 3 for s01, s02, and s03)
            if process_s010203:
                video = video.replace('RGB', 'Video')
                os.rename(video_file_path, os.path.join(eggnog_dataset_path, session_name, layout, video))  # video is renamed
                os.rename(video_file_path[:-4] + ".frames", os.path.join(eggnog_dataset_path, session_name, layout, video[:-4] + ".frames"))  # .frames is renamed
                video_file_path = os.path.join(eggnog_dataset_path, session_name, layout, video)
            
            #1
            xyz_skeleton_data, rgb_skeleton_data = load_skeleton_data_for_video(video_file_path)
            
            #2
            frame_time_dict = load_frames_file(video_file_path)
            
            #3
            npy_path = os.path.join(eggnog_dataset_path, session_name, layout)  # save npy files (in a video-named folder) in this folder
            
            # old version with no tracking taken into consideration in generation of gt (actually it is now), and no random augmentattion.
#             save_2d_keypoints_and_images_v1(video[:-4], video_file_path, npy_path, rgb_skeleton_data, frame_time_dict)
            
            # new version where augmentation is done and saved for every image. Also, tracking info is taken into consideration
            save_2d_keypoints_and_images_v2(video[:-4], video_file_path, npy_path, rgb_skeleton_data, frame_time_dict)
            
            

def test_skeleton_reader():
    video_file_path = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm/s04/part2_layout_p07/20151113_230303_00_Video.avi"
    xyz_skeleton_data, rgb_skeleton_data = load_skeleton_data_for_video(video_file_path)
    xyz_skeleton_data_v1, rgb_skeleton_data_v1 = load_skeleton_data_for_video_v1(video_file_path)
    
    index = -1
    print("xyz_skeleton_data, rgb_skeleton_data", xyz_skeleton_data.shape, rgb_skeleton_data.shape)
    print("rgb skeleton v2", rgb_skeleton_data[index])
    print("rgb skeleton v1", rgb_skeleton_data_v1[index])
    

def test_heatmap_gen():
    index_array = np.zeros((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, 2))
#     index_array = np.zeros((EggnogGlobalConfig.height, EggnogGlobalConfig.width, 2))
    for i in range(index_array.shape[0]):
        for j in range(index_array.shape[1]):
            index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
    
    ## 30/2 = 15; 16*4.5*8 = 540 20 => 720
    heatmap = get_heatmap(index_array, kpx_kpy_transformer([1920/2, 1080/2]), tracking_state=2)
#     heatmap = get_heatmap(index_array, [160, 120], tracking_state=2)
    
    skimage.io.imsave(os.path.join( "./test_imgs/test_15v2_tr2.jpg"), heatmap)
    
    
    # for image test_15_tr2.jpg, the GIMP display shows artifacts but the actual pixel values below are correct (relateive to each other)
    """
    print("(10, 17)", heatmap[10, 17])
    print("(14, 18)", heatmap[14, 18])
    print("(14, 17)", heatmap[14, 17])
    (10, 17) 0.0001590283861762805
    (14, 18) 0.03494073402815759  # hould be higher and it is, but image shows this pixel to be black
    (14, 17) 0.008708841628440166
    """
#     # show heatmap
#     print("heatmap testing")
#     plt.figure(0)
#     plt.imshow(heatmap, cmap=plt.get_cmap('gray'))
#     plt.show()


def test_paf_gen():
    index_array = np.zeros((EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor, EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor, 2))
#     index_array = np.zeros((EggnogGlobalConfig.height, EggnogGlobalConfig.width, 2))
    for i in range(index_array.shape[0]):
        for j in range(index_array.shape[1]):
            index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
    
    tracking_states = [0, 0]
    
    paf = get_pafx_pafy(index_array, # [20, 10], [10, 20],
                                        kp0xy=kpx_kpy_transformer([1920/3, 1080/3]), 
                                        kp1xy=kpx_kpy_transformer([1920/2, 1080/2]),
                                        tracking_states_pair = tracking_states
                                        )
    print(paf.shape, paf[:,:,0], paf[:,:,1])
    skimage.io.imsave(os.path.join( "./test_imgs/paf_1_tr00_x.jpg"), paf[:,:,0])
    skimage.io.imsave(os.path.join( "./test_imgs/paf_1_tr00_y.jpg"), paf[:,:,1])
    # these images are not displayed correctly because pafs have negative values as well

    
if __name__ == "__main__":
    print("reading videos and writing img240x320, paf30x40, and hm30x40...")
    
    """
    Usage: provide the session name as argument; the path of eggnog_dataset is needed to be set at the top as a global variable.
    """
    
    process_session(sys.argv[1])

#     test_skeleton_reader()
#     test_heatmap_gen()
#     test_paf_gen()
    
