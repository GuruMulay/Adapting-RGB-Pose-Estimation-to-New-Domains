import numpy as np

def ltr_parts(parts_dict):
    # when we flip image left parts became right parts and vice versa. This is the list of parts to exchange each other.
    leftParts  = [ parts_dict[p] for p in ["Lsho", "Lelb", "Lwri", "Lhip", "Lkne", "Lank", "Leye", "Lear"] ]
    rightParts = [ parts_dict[p] for p in ["Rsho", "Relb", "Rwri", "Rhip", "Rkne", "Rank", "Reye", "Rear"] ]
    return leftParts,rightParts


class EggnogGlobalConfig:

    original_w = 1920
    original_h = 1080
    
    width = 320
    height = 240

    stride = 8
    ground_truth_factor = 8  # 240/8=30 320/8=40
    
    kp_x_offset_half = 240  # remember we cropped 240 pixels from left and right side of the image
    kp_to_img_stride = 4.5  # factor by which kp should be scaled DOWN to make them valid in img space of width and height = (width, height)
    
    n_paf = 36
    n_hm = 20
    n_kp = 19
    n_axis = 2  # x and y as of now
    
    paf_pairs_indices = [[1, 14], [0, 1], [12, 0], [13, 0], 
                    [4, 14], [5, 4], [6, 5], [7, 6], [15, 7], [16, 6],
                    [8, 14], [9, 8], [10, 9], [11, 10], [17, 11], [18, 10],
                    [14, 2], [2, 3]
                    ]
    # useful when loading the numpy files in the generator
    paf_indices_xy = [p for p in range(2*len(paf_pairs_indices))]
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, ...., 34, 35]
    
    joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    left_joint_indices = [4, 5, 6, 7, 12, 15, 16]  # with 19 joints considered
    right_joint_indices = [8, 9, 10, 11, 13, 17, 18]  # with 19 joints considered
    
    # eggnog_to_coco_10_joints_mapping = np.array([0, 5, 6, 7, 2, 3, 4, 9, 8, 1])  # wrong
    # eggnog_to_coco_10_joints_mapping = np.array([0, 9, 4, 5, 6, 1, 2, 3, 8, 7])  # wrong right and left are swapped
    eggnog_to_coco_10_joints_mapping = np.array([0, 9, 1, 2, 3, 4, 5, 6, 7, 8])  # array indexing is faster https://stackoverflow.com/questions/26194389/numpy-rearrange-array-based-upon-index-array
    
    # haven't used following for eggnog
    parts = ["nose", "neck", "Rsho", "Relb", "Rwri", "Lsho", "Lelb", "Lwri", "Rhip", "Rkne", "Rank", "Lhip", "Lkne", "Lank", "Reye", "Leye", "Rear", "Lear"]
    num_parts = len(parts)
    parts_dict = dict(zip(parts, range(num_parts)))
    parts += ["background"]
    num_parts_with_background = len(parts)

    leftParts, rightParts = ltr_parts(parts_dict)

    # this numbers probably copied from matlab they are 1.. based not 0.. based
    limb_from = [2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16]
    limb_to = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    limbs_conn = zip(limb_from, limb_to)
    limbs_conn = [(fr - 1, to - 1) for (fr, to) in limbs_conn]

    paf_layers = 2*len(limbs_conn)
    heat_layers = num_parts
    num_layers = paf_layers + heat_layers + 1

    paf_start = 0
    heat_start = paf_layers
    bkg_start = paf_layers + heat_layers

    data_shape = (3, height, width)     # 3, 368, 368
    mask_shape = (height//stride, width//stride)  # 46, 46
    parts_shape = (num_layers, height//stride, width//stride)  # 57, 46, 46

    
class TransformationParams:

    target_dist = 1.0;  # 0.6; originally 0.6
    scale_prob = 0.5;   # TODO: this is actually scale unprobability, i.e. 1 = off, 0 = always, not sure if it is a bug or not
    scale_min = 0.9;  #  originally 0.5
    scale_max = 1.1;
    max_rotate_degree = 12.
    center_perterb_max = 40.  # x and y
    flip_prob = 0.5
    
    # added for eggnog
    alpha = 2.0
    limb_width = 1.0  # pixels  # for low res pafs
    # limb_width = 1.25*4  # pixels  # for high res pafs used for verification
    
    
    # not sure why sigma is greater than 1 and so large (7)
    #!# sigma = 7.
    #!# paf_thre = 8.  # it is original 1.0 * stride in this program


class RmpeCocoConfig:


    parts = ['nose', 'Leye', 'Reye', 'Lear', 'Rear', 'Lsho', 'Rsho', 'Lelb',
     'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank',
     'Rank']

    num_parts = len(parts)

    # for COCO neck is calculated like mean of 2 shoulders.
    parts_dict = dict(zip(parts, range(num_parts)))

    @staticmethod
    def convert(joints):

        result = np.zeros((joints.shape[0], RmpeGlobalConfig.num_parts, 3), dtype=np.float)
        result[:,:,2]=2.  # 2 - abstent, 1 visible, 0 - invisible

        for p in RmpeCocoConfig.parts:
            coco_id = RmpeCocoConfig.parts_dict[p]
            global_id = RmpeGlobalConfig.parts_dict[p]
            assert global_id!=1, "neck shouldn't be known yet"
            result[:,global_id,:]=joints[:,coco_id,:]

        neckG = RmpeGlobalConfig.parts_dict['neck']
        RshoC = RmpeCocoConfig.parts_dict['Rsho']
        LshoC = RmpeCocoConfig.parts_dict['Lsho']


        # no neck in coco database, we calculate it as averahe of shoulders
        # TODO: we use 0 - hidden, 1 visible, 2 absent - it is not coco values they processed by generate_hdf5
        both_shoulders_known = (joints[:, LshoC, 2]<2)  &  (joints[:, RshoC, 2]<2)
        result[both_shoulders_known, neckG, 0:2] = (joints[both_shoulders_known, RshoC, 0:2] +
                                                    joints[both_shoulders_known, LshoC, 0:2]) / 2
        result[both_shoulders_known, neckG, 2] = np.minimum(joints[both_shoulders_known, RshoC, 2],
                                                                 joints[both_shoulders_known, LshoC, 2])

        return result

class RpmeMPIIConfig:

    parts = ["HeadTop", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
             "RAnkle", "LHip", "LKnee", "LAnkle"]

    numparts = len(parts)

    #14 - Chest is calculated like "human center location provided by the annotated data"


    @staticmethod
    def convert(joints):
        raise "Not implemented"



# more information on keypoints mapping is here
# https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/issues/7


def check_layer_dictionary():

    dct = RmpeGlobalConfig.parts[:]
    dct = [None]*(RmpeGlobalConfig.num_layers-len(dct)) + dct

    for (i,(fr,to)) in enumerate(RmpeGlobalConfig.limbs_conn):
        name = "%s->%s" % (RmpeGlobalConfig.parts[fr], RmpeGlobalConfig.parts[to])
        print(i, name)
        x = i*2
        y = i*2+1

        assert dct[x] is None
        dct[x] = name + ":x"
        assert dct[y] is None
        dct[y] = name + ":y"

    print(dct)


if __name__ == "__main__":
    check_layer_dictionary()

