#!/usr/bin/env python

import numpy as np
from math import sqrt, isnan

from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig, TransformationParams
from py_eggnog_server.py_eggnog_transformer import check_if_out_of_the_frame


from sklearn.preprocessing import normalize
import skimage.io
from skimage.transform import resize, pyramid_reduce


class Heatmapper:

    def __init__(self, alpha=TransformationParams.alpha, limb_width=TransformationParams.limb_width):

        self.alpha = alpha
        self.limb_width = limb_width
        
        
        # cached common parameters which same for all iterations and all pictures
        self.gt_height = EggnogGlobalConfig.height//EggnogGlobalConfig.ground_truth_factor
        self.gt_width = EggnogGlobalConfig.width//EggnogGlobalConfig.ground_truth_factor
        
        self.paf_pairs = EggnogGlobalConfig.paf_pairs_indices
        self.verbose_aug = False
#         stride = RmpeGlobalConfig.stride
#         width = RmpeGlobalConfig.width/stride
#         height = RmpeGlobalConfig.height/stride


    def kpx_kpy_transformer(self, kp_list):
        """
        transform kps from 320x240 image space to 40x30 ground truth space
        """
        # transform [kpx kpy] from image space to ground truth space
        kpx_transformed = (kp_list[0])/EggnogGlobalConfig.ground_truth_factor
        kpy_transformed = (kp_list[1])/EggnogGlobalConfig.ground_truth_factor

        return [kpx_transformed, kpy_transformed]


    def kpx_kpy_transformer_v0(self, kp_list):
        """
        same as def kpx_kpy_transformer(kp_list): from read_videos_write....py
        used when dataset is generated meaning the images are extracted from videos and gt is generated
        transform kps from 1920x1080 to 320x240 space and then to 40x30 ground truth space
        """
        # for normal res pafs
        kpx_transformed = (kp_list[0]-EggnogGlobalConfig.kp_x_offset_half) / EggnogGlobalConfig.kp_to_img_stride/EggnogGlobalConfig.ground_truth_factor
        kpy_transformed = (kp_list[1])/EggnogGlobalConfig.kp_to_img_stride/EggnogGlobalConfig.ground_truth_factor

        return [kpx_transformed, kpy_transformed]
    

    def get_heatmap_v1(self, index_array, pxpy_list):
        # index_array (240/8, 320/8, 2), pxpy_list [px, py]
        kp_location_array = np.zeros(index_array.shape)
        assert(index_array.shape[0] > 0 and index_array.shape[1] > 0)

        if pxpy_list[0] is None:
            heatmap = kp_location_array[:,:,0]
        else:
            kp_location_array[:, :, 0] = pxpy_list[1]
            kp_location_array[:, :, 1] = pxpy_list[0]
            heatmap = np.exp((self.alpha)*-np.sqrt(np.sum(np.square(index_array - kp_location_array), axis=2)))
            heatmap /= np.max(heatmap)

    #     if np.sum(heat_map)>0:
    #         heat_map /= np.max(heat_map)

        return heatmap
    
    
    def get_heatmap(self, index_array, pxpy_list, tracking_state):
        # index_array (240/8, 320/8, 2), pxpy_list [px, py], tracking_state 0, 1, or 2
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
                heatmap = np.exp((self.alpha)*-np.sqrt(np.sum(np.square(index_array - kp_location_array), axis=2)))
                heatmap /= np.max(heatmap)

                if tracking_state == 1:  # joint is inferred, so return grey gaussian instead of white one?
                    ### print("inferred kp")
                    heatmap /= 2  # using this the white peak of gaussian is reduced in its magnitude by a factor of 2

        return heatmap


    
    def get_pafx_pafy_v1(self, index_array, kp0xy, kp1xy):
        """
        kp0xy, kp1xy: lists [pixel_x, pixel_y] for kp0 and kp1
        """
        
        assert(index_array.shape[0] > 0 and index_array.shape[1] > 0)

        paf_array = np.zeros(index_array.shape)  # stores calculated pafs (values between -1 to 1)
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
                if (0 <= np.dot(v_unit, [p_vector_array[r,c,0], p_vector_array[r,c,1]]) <= limb_length) and (np.abs(np.dot(v_perpendicular, [p_vector_array[r,c,0], p_vector_array[r,c,1]])) <= self.limb_width):
                    paf_array[r, c, 1] = v_unit[0] # y component of the expected vector is assigned to the 1st channel so that we append PAFs in x and then y order
                    paf_array[r, c, 0] = v_unit[1] # x component of the expected vector is assigned to the 0th channel so that we append PAFs in x and then y order
    #                 print("x, y", v_unit[1], v_unit[0])
    #             else:
    #                 print("should assign zero")
        # return paf so that x slice is first (index 0) and the y slice is second (index 1)

        return paf_array

    
    def get_pafx_pafy(self, index_array, kp0xy, kp1xy, tracking_states_pair):
        """
        kp0xy, kp1xy: lists [pixel_x, pixel_y] for kp0 and kp1
        tracking_states_pair = [0, 0] or [1, 0] etc
        """
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
                if (0 <= np.dot(v_unit, [p_vector_array[r,c,0], p_vector_array[r,c,1]]) <= limb_length) and (np.abs(np.dot(v_perpendicular, [p_vector_array[r,c,0], p_vector_array[r,c,1]])) <= self.limb_width):
                    paf_array[r, c, 1] = v_unit[0] # y component of the expected vector is assigned to the 1st channel so that we append PAFs in x and then y order
                    paf_array[r, c, 0] = v_unit[1] # x component of the expected vector is assigned to the 0th channel so that we append PAFs in x and then y order
    #                 print("x, y", v_unit[1], v_unit[0])
    #             else:
    #                 print("should assign zero")
        # return paf so that x slice is first (index 0) and the y slice is second (index 1)

        return paf_array

    
    def get_pafs_and_hms_heatmaps_additional_joints_version(self, sk_keypoints, sk_kp_tracking_info):
        """
        _additional_joints_version: added additional joints for averaged hand joints, also added three versions of background heatmaps (see notes from sheet 2).
        sk_keypoints: (38,) shaped keypoints with alternate x and y corrdinates for 19 joints; # these kp are in the image space (240x320)
        kp_tracking_info: 0 means UNTRACKED or out of frame
        """
        
        # 3 from eggnog_preprocessing/preprocessing/read_videos_write_img_paf_hm.py
        # print("sk_kp shape =", sk_keypoints.shape)  # (38, )
        
        # for 20 (actually 19 + background) heatmaps =====================================
        for kpn in range(sk_keypoints.shape[0]//2):
            kpx = sk_keypoints[2*kpn]
            kpy = sk_keypoints[2*kpn+1]  # print(kpx, kpy)
            tracking_state = sk_kp_tracking_info[kpn]
            
            index_array = np.zeros((self.gt_height, self.gt_width, 2))
            for i in range(index_array.shape[0]):
                for j in range(index_array.shape[1]):
                    index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                
            if kpn == 0:
                heatmap = self.get_heatmap(index_array, self.kpx_kpy_transformer([kpx, kpy]), tracking_state)
            else:
                heatmap = np.dstack(( heatmap, self.get_heatmap(index_array, self.kpx_kpy_transformer([kpx, kpy]), tracking_state) ))
            # print("heatmap.shape =", heatmap.shape)

        tracking_info_additional = []  # to store tracking info of additional indices for background use '3'
        sk_kp_additional = []  # to store x, y for new two joints (avg of left and right) use 0, 0 for background

        # generate background heatmap
        maxed_heatmap = np.max(heatmap[:,:,:], axis=2)  # print("maxed_heatmap.shape = ", maxed_heatmap.shape)
        heatmap = np.dstack((heatmap, 1 - maxed_heatmap))

        tracking_info_additional.append(3)  # index19
        sk_kp_additional.append(0)  # for x
        sk_kp_additional.append(0)  # for y

        if self.verbose_aug: print("heatmapper ======================")
        # #
        ### July 26th, 2018 (Added general purpose GT for multiple types of experiments)
        # generate background heatmap for the common joints with coco
        if self.verbose_aug: print("common_joints_with_coco", EggnogGlobalConfig.common_joints_with_coco)
        maxed_heatmap = np.max(heatmap[:, :, EggnogGlobalConfig.common_joints_with_coco], axis=2)
        # index 20
        heatmap = np.dstack((heatmap, 1 - maxed_heatmap))
        tracking_info_additional.append(3)  # index20
        sk_kp_additional.append(0)  # for x
        sk_kp_additional.append(0)  # for y

        # generate background heatmap for the case where additional_spine joints are added (0, 1, 2 joints)
        bk_hm_indices = [x for x in EggnogGlobalConfig.all_19_joint_indices if x not in EggnogGlobalConfig.left_hand_joint_indices + EggnogGlobalConfig.right_hand_joint_indices]
        if self.verbose_aug: print("bk_hm_indices after adding 0, 1, 2", bk_hm_indices)
        if self.verbose_aug: print("common_joints_with_coco + additional_spine_indices", EggnogGlobalConfig.common_joints_with_coco + EggnogGlobalConfig.additional_spine_indices)
        assert (set(bk_hm_indices) == set(EggnogGlobalConfig.common_joints_with_coco + EggnogGlobalConfig.additional_spine_indices))

        maxed_heatmap = np.max(heatmap[:, :, bk_hm_indices], axis=2)
        # index 21
        heatmap = np.dstack((heatmap, 1 - maxed_heatmap))
        tracking_info_additional.append(3)  # index21
        sk_kp_additional.append(0)  # for x
        sk_kp_additional.append(0)  # for y
        
        
        # add average of 3 left hand joints and then 3 right hand joints at index 22 and 23 (after index 0-19 and 20, 21)
        # left 3 joints
        index_array = np.zeros((EggnogGlobalConfig.height // EggnogGlobalConfig.ground_truth_factor,
                                EggnogGlobalConfig.width // EggnogGlobalConfig.ground_truth_factor, 2))
        for i in range(index_array.shape[0]):
            for j in range(index_array.shape[1]):
                index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
        kpx_indices = [2 * ind for ind in EggnogGlobalConfig.left_hand_joint_indices]
        kpy_indices = [2 * ind + 1 for ind in EggnogGlobalConfig.left_hand_joint_indices]
        if self.verbose_aug: print("kpx_indices, kpy_indices", kpx_indices, kpy_indices)

        kpx = np.mean(sk_keypoints[kpx_indices])
        kpy = np.mean(sk_keypoints[kpy_indices])
        if self.verbose_aug: print("mean of left hand joints x and y", kpx, kpy)

        # kp_tracking_info is updated to UNTRACKED (0) if kp falls beyond the image w or h
        if check_if_out_of_the_frame(kpx, kpy):  # passing x and y
            tracking_state = 0
            if self.verbose_aug: print("transformed kp is out of the frame, setting tracking info to UNTRACKED (0) for mean kp of hand joints")
        else:
            tracking_state = int(np.mean(sk_kp_tracking_info[EggnogGlobalConfig.left_hand_joint_indices]))
        if self.verbose_aug: print("tracking_state", tracking_state)

        tracking_info_additional.append(tracking_state)  # index22
        sk_kp_additional.append(kpx)  # for x
        sk_kp_additional.append(kpy)  # for y

        # index 22
        heatmap = np.dstack((heatmap, self.get_heatmap(index_array, self.kpx_kpy_transformer([kpx, kpy]), tracking_state)))

        # right 3 joints
        index_array = np.zeros((EggnogGlobalConfig.height // EggnogGlobalConfig.ground_truth_factor,
                                EggnogGlobalConfig.width // EggnogGlobalConfig.ground_truth_factor, 2))
        for i in range(index_array.shape[0]):
            for j in range(index_array.shape[1]):
                index_array[i][j] = [i,j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
        kpx_indices = [2 * ind for ind in EggnogGlobalConfig.right_hand_joint_indices]
        kpy_indices = [2 * ind + 1 for ind in EggnogGlobalConfig.right_hand_joint_indices]
        if self.verbose_aug: print("kpx_indices, kpy_indices", kpx_indices, kpy_indices)

        kpx = np.mean(sk_keypoints[kpx_indices])
        kpy = np.mean(sk_keypoints[kpy_indices])
        if self.verbose_aug: print("mean of right hand joints x and y", kpx, kpy)

        # kp_tracking_info is updated to UNTRACKED (0) if kp falls beyond the image w or h
        if check_if_out_of_the_frame(kpx, kpy):  # passing x and y
            tracking_state = 0
            if self.verbose_aug: print("transformed kp is out of the frame, setting tracking info to UNTRACKED (0) for mean kp of hand joints")
        else:
            tracking_state = int(np.mean(sk_kp_tracking_info[EggnogGlobalConfig.right_hand_joint_indices]))
        if self.verbose_aug: print("tracking_state", tracking_state)

        tracking_info_additional.append(tracking_state)  # index23
        sk_kp_additional.append(kpx)  # for x
        sk_kp_additional.append(kpy)  # for y

        # index 23
        heatmap = np.dstack((heatmap, self.get_heatmap(index_array, self.kpx_kpy_transformer([kpx, kpy]), tracking_state)))

        # generate background heatmap for the case where hand joints are averaged
        bk_hm_indices = [x for x in EggnogGlobalConfig.all_19_joint_indices if
                         x not in EggnogGlobalConfig.left_hand_joint_indices + EggnogGlobalConfig.right_hand_joint_indices] + [EggnogGlobalConfig.avg_l_idx, EggnogGlobalConfig.avg_r_idx]
        if self.verbose_aug: print("bk_hm_indices to get index 24 for avg l and r hand joints", bk_hm_indices)
        maxed_heatmap = np.max(heatmap[:, :, bk_hm_indices], axis=2)
        # index 24
        heatmap = np.dstack((heatmap, 1 - maxed_heatmap))

        tracking_info_additional.append(3)  # index24
        sk_kp_additional.append(0)  # for x
        sk_kp_additional.append(0)  # for y


        if self.verbose_aug: print("tracking_info_additional", tracking_info_additional)
        sk_kp_tracking_info_additional = np.append(sk_kp_tracking_info, tracking_info_additional)
        if self.verbose_aug: print("sk_kp_tracking_info_additional", sk_kp_tracking_info_additional, len(sk_kp_tracking_info_additional))

        if self.verbose_aug: print("sk_kp_additional", sk_kp_additional)
        sk_keypoints_additional = np.append(sk_keypoints, sk_kp_additional)
        if self.verbose_aug: print("sk_keypoints_additional", sk_keypoints_additional, len(sk_keypoints_additional))


        # print("final heatmap.shape =", heatmap.shape)
        # np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_heatmap30x40.npy"), heatmap)
            
            
        # for 18x2 PAFs =====================================
        for n, pair in enumerate(self.paf_pairs):
            # print("writing paf for index", n, pair)
            index_array = np.zeros((self.gt_height, self.gt_width, 2))
            for i in range(index_array.shape[0]):
                for j in range(index_array.shape[1]):
                    index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                        
            tracking_states = [sk_kp_tracking_info_additional[pair[0]], sk_kp_tracking_info_additional[pair[1]]]
            
            if n == 0:
                paf = self.get_pafx_pafy(index_array, 
                                    kp0xy=self.kpx_kpy_transformer([sk_keypoints_additional[2*pair[0]], sk_keypoints_additional[2*pair[0]+1]]),
                                    kp1xy=self.kpx_kpy_transformer([sk_keypoints_additional[2*pair[1]], sk_keypoints_additional[2*pair[1]+1]]),
                                    tracking_states_pair=tracking_states
                                    )
            else:
                paf = np.dstack(( paf,  self.get_pafx_pafy(index_array, 
                                kp0xy=self.kpx_kpy_transformer([sk_keypoints_additional[2*pair[0]], sk_keypoints_additional[2*pair[0]+1]]),
                                kp1xy=self.kpx_kpy_transformer([sk_keypoints_additional[2*pair[1]], sk_keypoints_additional[2*pair[1]+1]]),
                                tracking_states_pair=tracking_states
                                )
                                ))
            # print("paf.shape =", paf.shape)

                    
        # print("final paf.shape =========================", paf.shape)
        # np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_paf30x40.npy"), paf)
        
        return paf, heatmap
    
    
    
    def get_pafs_and_hms_heatmaps(self, sk_keypoints, sk_kp_tracking_info):
        """
        sk_keypoints: (38,) shaped keypoints with alternate x and y corrdinates for 19 joints; # these kp are in the image space (240x320)
        kp_tracking_info: 0 means UNTRACKED or out of frame
        """
        
        # 3 from eggnog_preprocessing/preprocessing/read_videos_write_img_paf_hm.py
        # print("sk_kp shape =", sk_keypoints.shape)  # (38, )
        
        # for 20 (actually 19 + background) heatmaps =====================================
        for kpn in range(sk_keypoints.shape[0]//2):
            kpx = sk_keypoints[2*kpn]
            kpy = sk_keypoints[2*kpn+1]  # print(kpx, kpy)
            tracking_state = sk_kp_tracking_info[kpn]
            
            index_array = np.zeros((self.gt_height, self.gt_width, 2))
            for i in range(index_array.shape[0]):
                for j in range(index_array.shape[1]):
                    index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                
            if kpn == 0:
                heatmap = self.get_heatmap(index_array, self.kpx_kpy_transformer([kpx, kpy]), tracking_state)   # transform from image space to ground truth space
            else:
                heatmap = np.dstack(( heatmap, self.get_heatmap(index_array, self.kpx_kpy_transformer([kpx, kpy]), tracking_state) ))
            # print("heatmap.shape =", heatmap.shape)
            
        # generate background heatmap
        maxed_heatmap = np.max(heatmap[:,:,:], axis=2)  # print("maxed_heatmap.shape = ", maxed_heatmap.shape)
            
        heatmap = np.dstack((heatmap, 1 - maxed_heatmap))
        # print("final heatmap.shape =", heatmap.shape)
        # np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_heatmap30x40.npy"), heatmap)
            
            
        # for 18x2 PAFs =====================================
        for n, pair in enumerate(self.paf_pairs):
            # print("writing paf for index", n, pair)
            index_array = np.zeros((self.gt_height, self.gt_width, 2))
            for i in range(index_array.shape[0]):
                for j in range(index_array.shape[1]):
                        index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                        
            tracking_states = [sk_kp_tracking_info[pair[0]], sk_kp_tracking_info[pair[1]]]
            
            if n == 0:
                paf = self.get_pafx_pafy(index_array, 
                                    kp0xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                    kp1xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]),
                                    tracking_states_pair=tracking_states
                                    )
            else:
                paf = np.dstack(( paf,  self.get_pafx_pafy(index_array, 
                                kp0xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                kp1xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]),
                                tracking_states_pair=tracking_states
                                )
                                ))
            # print("paf.shape =", paf.shape)

                    
        # print("final paf.shape =========================", paf.shape)
        # np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_paf30x40.npy"), paf)
        
        return paf, heatmap
    
    
    def get_pafs_and_hms_heatmaps_v1(self, sk_keypoints):
        """
        sk_keypoints: (38,) shaped keypoints with alternate x and y coordinates for 19 joints; # these kp are in the image space (240x320)
        """
        
        # 3 from eggnog_preprocessing/preprocessing/read_videos_write_img_paf_hm.py
        # print("sk_kp shape =", sk_keypoints.shape)  # (38, )
        
        # for 20 (actually 19 + background) heatmaps =====================================
        for kpn in range(sk_keypoints.shape[0]//2):
            kpx = sk_keypoints[2*kpn]
            kpy = sk_keypoints[2*kpn+1]  # print(kpx, kpy)
                
            index_array = np.zeros((self.gt_height, self.gt_width, 2))
            for i in range(index_array.shape[0]):
                for j in range(index_array.shape[1]):
                    index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                
            if kpn == 0:
                heatmap = self.get_heatmap_v1(index_array, self.kpx_kpy_transformer([kpx, kpy]))   # transform from image space to ground truth space
            else:
                heatmap = np.dstack(( heatmap, self.get_heatmap_v1(index_array, self.kpx_kpy_transformer([kpx, kpy])) ))
            # print("heatmap.shape =", heatmap.shape)
            
        # generate background heatmap
        maxed_heatmap = np.max(heatmap[:,:,:], axis=2)  # print("maxed_heatmap.shape = ", maxed_heatmap.shape)
            
        heatmap = np.dstack((heatmap, 1 - maxed_heatmap))
        # print("final heatmap.shape =", heatmap.shape)
        # np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_heatmap30x40.npy"), heatmap)
            
            
        # for 18x2 PAFs =====================================
        for n, pair in enumerate(self.paf_pairs):
            # print("writing paf for index", n, pair)
            index_array = np.zeros((self.gt_height, self.gt_width, 2))
            for i in range(index_array.shape[0]):
                for j in range(index_array.shape[1]):
                        index_array[i][j] = [i, j]  # height (y), width (x) => index_array[:,:,0] = y pixel coordinate and index_array[:,:,1] = x
                
            if n == 0:
                paf = self.get_pafx_pafy_v1(index_array, 
                                    kp0xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                    kp1xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]))
            else:
                paf = np.dstack(( paf,  self.get_pafx_pafy_v1(index_array, 
                                kp0xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[0]], sk_keypoints[2*pair[0]+1]]), 
                                kp1xy=self.kpx_kpy_transformer([sk_keypoints[2*pair[1]], sk_keypoints[2*pair[1]+1]]))
                                ))
            # print("paf.shape =", paf.shape)

                    
        # print("final paf.shape =========================", paf.shape)
        # np.save(os.path.join(save_dir, video_name + "_vfr_" + str(k) + "_skfr_" + str(nearest_idx) + "_paf30x40.npy"), paf)
        
        return paf, heatmap
    
        
#     def create_heatmaps(self, joints, mask):

#         heatmaps = np.zeros(RmpeGlobalConfig.parts_shape, dtype=np.float)

#         self.put_joints(heatmaps, joints)
#         sl = slice(RmpeGlobalConfig.heat_start, RmpeGlobalConfig.heat_start + RmpeGlobalConfig.heat_layers)
#         heatmaps[RmpeGlobalConfig.bkg_start] = 1. - np.amax(heatmaps[sl,:,:], axis=0)

#         self.put_limbs(heatmaps, joints)

#         heatmaps *= mask

#         return heatmaps


#     def put_gaussian_maps(self, heatmaps, layer, joints):

#         # actually exp(a+b) = exp(a)*exp(b), lets use it calculating 2d exponent, it could just be calculated by

#         for i in range(joints.shape[0]):

#             exp_x = np.exp(-(self.grid_x-joints[i,0])**2/self.double_sigma2)
#             exp_y = np.exp(-(self.grid_y-joints[i,1])**2/self.double_sigma2)

#             exp = np.outer(exp_y, exp_x)

#             # note this is correct way of combination - min(sum(...),1.0) as was in C++ code is incorrect
#             # https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/118
#             heatmaps[RmpeGlobalConfig.heat_start + layer, :, :] = np.maximum(heatmaps[RmpeGlobalConfig.heat_start + layer, :, :], exp)

#     def put_joints(self, heatmaps, joints):

#         for i in range(RmpeGlobalConfig.num_parts):
#             visible = joints[:,i,2] < 2
#             self.put_gaussian_maps(heatmaps, i, joints[visible, i, 0:2])


#     def put_vector_maps(self, heatmaps, layerX, layerY, joint_from, joint_to):

#         count = np.zeros(heatmaps.shape[1:], dtype=np.int)

#         for i in range(joint_from.shape[0]):
#             (x1, y1) = joint_from[i]
#             (x2, y2) = joint_to[i]

#             dx = x2-x1
#             dy = y2-y1
#             dnorm = sqrt(dx*dx + dy*dy)

#             if dnorm==0:  # we get nan here sometimes, it's kills NN
#                 # TODO: handle it better. probably we should add zero paf, centered paf, or skip this completely
#                 print("Parts are too close to each other. Length is zero. Skipping")
#                 continue

#             dx = dx / dnorm
#             dy = dy / dnorm

#             assert not isnan(dx) and not isnan(dy), "dnorm is zero, wtf"

#             min_sx, max_sx = (x1, x2) if x1 < x2 else (x2, x1)
#             min_sy, max_sy = (y1, y2) if y1 < y2 else (y2, y1)

#             min_sx = int(round((min_sx - self.thre) / RmpeGlobalConfig.stride))
#             min_sy = int(round((min_sy - self.thre) / RmpeGlobalConfig.stride))
#             max_sx = int(round((max_sx + self.thre) / RmpeGlobalConfig.stride))
#             max_sy = int(round((max_sy + self.thre) / RmpeGlobalConfig.stride))

#             # check PAF off screen. do not really need to do it with max>grid size
#             if max_sy < 0:
#                 continue

#             if max_sx < 0:
#                 continue

#             if min_sx < 0:
#                 min_sx = 0

#             if min_sy < 0:
#                 min_sy = 0

#             #TODO: check it again
#             slice_x = slice(min_sx, max_sx) # + 1     this mask is not only speed up but crops paf really. This copied from original code
#             slice_y = slice(min_sy, max_sy) # + 1     int g_y = min_y; g_y < max_y; g_y++ -- note strict <

#             dist = distances(self.X[slice_y,slice_x], self.Y[slice_y,slice_x], x1, y1, x2, y2)
#             dist = dist <= self.thre

#             # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
#             heatmaps[layerX, slice_y, slice_x][dist] = (dist * dx)[dist]  # += dist * dx
#             heatmaps[layerY, slice_y, slice_x][dist] = (dist * dy)[dist] # += dist * dy
#             count[slice_y, slice_x][dist] += 1

#         # TODO: averaging by pafs mentioned in the paper but never worked in C++ augmentation code
#         # heatmaps[layerX, :, :][count > 0] /= count[count > 0]
#         # heatmaps[layerY, :, :][count > 0] /= count[count > 0]

#     def put_limbs(self, heatmaps, joints):

#         for (i,(fr,to)) in enumerate(RmpeGlobalConfig.limbs_conn):


#             visible_from = joints[:,fr,2] < 2
#             visible_to = joints[:,to, 2] < 2
#             visible = visible_from & visible_to

#             layerX, layerY = (RmpeGlobalConfig.paf_start + i*2, RmpeGlobalConfig.paf_start + i*2 + 1)
#             self.put_vector_maps(heatmaps, layerX, layerY, joints[visible, fr, 0:2], joints[visible, to, 0:2])



# def test():

#     hm = Heatmapper()
#     d = distances(hm.X, hm.Y, 100, 100, 50, 150)
#     print(d < 8.)

# if __name__ == "__main__":
#     np.set_printoptions(precision=1, linewidth=1000, suppress=True, threshold=100000)
#     test()

