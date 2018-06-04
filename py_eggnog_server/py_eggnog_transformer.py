import numpy as np
from math import cos, sin, pi
import cv2
import random

from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig, TransformationParams


def keypoint_transform_to_240x320_image(kp, flip):
    """
    Transforms kps (which are in 1080x1920 space) to 240x320 space. Note that this is not the ground truth space 30x40.
    This flip is unnecessary because matrix M already takes care of flipping. So keep flip=False always.
    """
    
    # print("before 240x320 tx", kp.shape, kp)
    for k in range(len(kp)):
        if k%2 == 0:  # x axis
            if flip:
                kp[k] = EggnogGlobalConfig.width - ((kp[k] - EggnogGlobalConfig.kp_x_offset_half)/EggnogGlobalConfig.kp_to_img_stride)  # 320 - x
            else:
                kp[k] = (kp[k]-EggnogGlobalConfig.kp_x_offset_half)/EggnogGlobalConfig.kp_to_img_stride
                
        else:  # y axis
            kp[k] = kp[k]/EggnogGlobalConfig.kp_to_img_stride
        
    return kp
    


class AugmentSelection:

    def __init__(self, flip=False, degree = 0.0, crop = (0,0), scale = 1.0):
        self.flip = flip
        self.degree = degree  #rotate
        self.crop = crop  #shift actually
        self.scale = scale
        
        
    def print_aug_params(self,):
        print("self.flip", self.flip)
        print("self.degree", self.degree)
        print("self.crop", self.crop)
        print("self.scale", self.scale)

       
    @staticmethod
    def random():
        flip = random.uniform(0.,1.) > TransformationParams.flip_prob
        # flip = False
        degree = random.uniform(-1.,1.) * TransformationParams.max_rotate_degree
        scale = (TransformationParams.scale_max - TransformationParams.scale_min)*random.uniform(0.,1.)+TransformationParams.scale_min \
            if random.uniform(0.,1.) > TransformationParams.scale_prob else 1. # TODO: see 'scale improbability' TODO above
        x_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);
        y_offset = int(random.uniform(-1.,1.) * TransformationParams.center_perterb_max);
#         print("scale in random() = ", scale)  # currently it's always 1 => no scaling
#         print("random aug params ====", flip, degree, (x_offset,y_offset), scale)
        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    
    @staticmethod
    def unrandom():
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0.
        y_offset = 0.

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    
    def affine(self, center, scale_self):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards


        A = self.scale * cos(self.degree / 180. * pi )
        B = self.scale * sin(self.degree / 180. * pi )

        scale_size = TransformationParams.target_dist / scale_self * self.scale
        ### print("scale_size", scale_size)
        
        (width, height) = center
        ### print("center width, height", width, height)
        
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array( [[ 1., 0., -center_x],
                                 [ 0., 1., -center_y ],
                                 [ 0., 0., 1. ]] )

        rotate = np.array( [[ A, B, 0 ],
                           [ -B, A, 0 ],
                           [  0, 0, 1. ] ])

        scale = np.array( [[ scale_size, 0, 0 ],
                           [ 0, scale_size, 0 ],
                           [  0, 0, 1. ] ])

        flip = np.array( [[ -1 if self.flip else 1., 0., 0. ],
                          [ 0., 1., 0. ],
                          [ 0., 0., 1. ]] )

        center2center = np.array( [[ 1., 0., EggnogGlobalConfig.width//2],
                                   [ 0., 1., EggnogGlobalConfig.height//2 ],
                                   [ 0., 0., 1. ]] )

        # order of combination is reversed
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)

        return combined[0:2]

    
class Transformer:

    @staticmethod
    def transform(img, kp, kp_tracking_info, aug=AugmentSelection.random()):
#         print("in class transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)
#         print("img before", img[200][200][:])
    
#         # TODO: need to understand this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
#         print(img.shape, type(img), img.dtype, img.shape)

        assert np.isnan(kp).any() == False  # check if all elements are not nan
        kp_center_x = (kp[0] + kp[2] + kp[28])/3   # sum of spineshoulder, spinemid, and spinebase
        kp_center_y = (kp[1] + kp[3] + kp[29])/3   # sum of spineshoulder, spinemid, and spinebase
        ### print("kp center x = kp[0], kp[2], kp[28]", kp[0], kp[2], kp[28])  # 992.4157 991.5563 991.6354
        ### print("kp center y = kp[1], kp[3], kp[29]", kp[1], kp[3], kp[29])  # 633.3717 490.9328 355.1452
        ### print("kp_center_x y", kp_center_x, kp_center_y)
    
        M = aug.affine([(kp_center_x-EggnogGlobalConfig.kp_x_offset_half)/EggnogGlobalConfig.kp_to_img_stride, kp_center_y/EggnogGlobalConfig.kp_to_img_stride], 1)  # normalized height of the person in image! (assumed, need to change)
#         print("kp based center = ", (kp_center_x-EggnogGlobalConfig.kp_x_offset_half)/EggnogGlobalConfig.kp_to_img_stride, kp_center_y/EggnogGlobalConfig.kp_to_img_stride)
#         M = aug.affine([160, 120], 0.65)  # normalized height of the person in image! (assumed, need to change) 


        # before transforming the keypoints by affine transform M, we need to bring them in the 240x320 image space from the original 1080x1920 space.
#         kp = keypoint_transform_to_240x320_image(kp, aug.flip)  # Incorrect. This flip is unnecessary because matrix M already takes care of flipping. So keep flip=False always.
        kp = keypoint_transform_to_240x320_image(kp, flip=False)
#         print("after 240x320 tx", kp.shape, kp)
        
        # apply affine transform to kps
        kp_original = np.append(kp.reshape((1, EggnogGlobalConfig.n_kp, EggnogGlobalConfig.n_axis)), 
                               np.ones((1, EggnogGlobalConfig.n_kp, 1)),
                               axis = 2)  # third column is made all 1s because we want to multiply by affine mat M
        
        kp_converted = np.matmul(M, kp_original.transpose([0,2,1])).transpose([0,2,1])
        
        ######### TODO
        # kp_tracking_info is updated if kp falls beyond the image w or h
        #########
        
#         print("img, M shapes", img.shape, M, M.shape)  # (240, 320, 3) (2, 3)
#         cv2.imshow("before transform", img)
#         cv2.waitKey(0)
        
        img = cv2.warpAffine(img, M, (EggnogGlobalConfig.width, EggnogGlobalConfig.height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        

        return img, kp_converted.reshape((EggnogGlobalConfig.n_kp*2,)), kp_tracking_info


    @staticmethod
    def transform_v1(img, kp, aug=AugmentSelection.random()):
#         print("in class transform", img.shape, label_paf.shape, label_hm.shape, kp.shape)
#         print("img before", img[200][200][:])
    
#         # warp picture and mask
#         M = aug.affine(meta['objpos'][0], meta['scale_provided'][0])

#         # TODO: need to understand this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
#         print(img.shape, type(img), img.dtype, img.shape)


        assert np.isnan(kp).any() == False  # check if all elements are not nan
        kp_center_x = (kp[0] + kp[2] + kp[28])/3   # sum of spineshoulder, spinemid, and spinebase
        kp_center_y = (kp[1] + kp[3] + kp[29])/3   # sum of spineshoulder, spinemid, and spinebase
#         print("kp center x = kp[0], kp[2], kp[28]", kp[0], kp[2], kp[28])  # 992.4157 991.5563 991.6354
#         print("kp center y = kp[1], kp[3], kp[29]", kp[1], kp[3], kp[29])  # 633.3717 490.9328 355.1452
#         print("kp_center_x y", kp_center_x, kp_center_y)
    
        M = aug.affine([(kp_center_x-EggnogGlobalConfig.kp_x_offset_half)/EggnogGlobalConfig.kp_to_img_stride, kp_center_y/EggnogGlobalConfig.kp_to_img_stride], 1)  # normalized height of the person in image! (assumed, need to change)
#         print("kp based center = ", (kp_center_x-EggnogGlobalConfig.kp_x_offset_half)/EggnogGlobalConfig.kp_to_img_stride, kp_center_y/EggnogGlobalConfig.kp_to_img_stride)
#         M = aug.affine([160, 120], 0.65)  # normalized height of the person in image! (assumed, need to change) 

        
#         img = cv2.warpAffine(img, M, (EggnogGlobalConfig.height, EggnogGlobalConfig.width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))  # ValueError: could not broadcast input array from shape (320,240,3) into shape (240,320,3)
        
        # before transforming the keypoints by affine transform M, we need to bring them in the 240x320 image space from the original 1080x1920 space.
#         print("before 240x320 tx", kp.shape, kp)
#         kp = keypoint_transform_to_240x320_image(kp, aug.flip)  # Incorrect. This flip is unnecessary because matrix M already takes care of flipping. So keep flip=False always.
        kp = keypoint_transform_to_240x320_image(kp, flip=False)
#         print("after 240x320 tx", kp.shape, kp)
        
        # apply affine transform to kps
        kp_original = np.append(kp.reshape((1, EggnogGlobalConfig.n_kp, EggnogGlobalConfig.n_axis)), 
                               np.ones((1, EggnogGlobalConfig.n_kp, 1)),
                               axis = 2)  # third column is made all 1s because we want to multiply by affine mat M
        
        kp_converted = np.matmul(M, kp_original.transpose([0,2,1])).transpose([0,2,1])
                    
        
#         print("img, M shapes", img.shape, M, M.shape)  # (240, 320, 3) (2, 3)
#         cv2.imshow("before transform", img)
#         cv2.waitKey(0)
        
        img = cv2.warpAffine(img, M, (EggnogGlobalConfig.width, EggnogGlobalConfig.height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        
        
#         # this is incorrect because the M calculated for image cannot be use to transform the ground truth maps. M was calculated on 240x320 image, but ground truth is 30x40 image. 
#         for i in range(EggnogGlobalConfig.n_hm):
#             label_hm[:, :, i] = cv2.warpAffine(label_hm[:, :, i], M, (EggnogGlobalConfig.width/EggnogGlobalConfig.stride, EggnogGlobalConfig.height/EggnogGlobalConfig.stride), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#             ##### !!!!! NEED to figure out the DEFAULT for pixels after transform value which is zero if no joint is present
        
#         for i in range(EggnogGlobalConfig.n_paf):
#             label_paf[:, :, i] = cv2.warpAffine(label_paf[:, :, i], M, (EggnogGlobalConfig.width/EggnogGlobalConfig.stride, EggnogGlobalConfig.height/EggnogGlobalConfig.stride), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
#             ##### !!!!! NEED to figure out the DEFAULT for pixels after transform value which is zero if no joint is present




#         mask = cv2.warpAffine(mask, M, (EggnogGlobalConfig.height, EggnogGlobalConfig.width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
#         mask = cv2.resize(mask, EggnogGlobalConfig.mask_shape, interpolation=cv2.INTER_CUBIC)  # TODO: should be combined with warp for speed
#         #_, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
#         #assert np.all((mask == 0) | (mask == 255)), "Interpolation of mask should be thresholded only 0 or 255\n" + str(mask)
#         mask = mask.astype(np.float) / 255.



#         # warp key points
#         #TODO: joint could be cropped by augmentation, in this case we should mark it as invisible.
#         #update: may be we don't need it actually, original code removed part sliced more than half totally, may be we should keep it
#         original_points = meta['joints'].copy()
#         original_points[:,:,2]=1  # we reuse 3rd column in completely different way here, it is hack
#         converted_points = np.matmul(M, original_points.transpose([0,2,1])).transpose([0,2,1])
#         meta['joints'][:,:,0:2]=converted_points

        

#         # we just made image flip, i.e. right leg just became left leg, and vice versa

#         if aug.flip:
#             tmpLeft = meta['joints'][:, EggnogGlobalConfig.leftParts, :]
#             tmpRight = meta['joints'][:, EggnogGlobalConfig.rightParts, :]
#             meta['joints'][:, EggnogGlobalConfig.leftParts, :] = tmpRight
#             meta['joints'][:, EggnogGlobalConfig.rightParts, :] = tmpLeft
        
#         print("returning transformed data and labels...")

#         print("img after", img[100][100][1])
#         print("img all grey???", (img==127).all())  # all are 127 pixels
        return img, kp_converted.reshape((EggnogGlobalConfig.n_kp*2,))

