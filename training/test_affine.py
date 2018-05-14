# test on /s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/s07_1video
# transform only works on one layout folder at a time

import sys
import os
sys.path.append("..")
import numpy as np
from math import cos, sin, pi

import cv2
import skimage.io
from matplotlib import pyplot as plt

img_width = 320
img_height = 240
    
ID = "/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/s07_1video/part1_layout_p14/20151116_230338_00_Video/20151116_230338_00_Video_vfr_0_skfr_0_240x320.jpg"


# aug parameters # write random paramneters
# degree = 0
# scale = 1
# crop = (0, 0)
# flip = False

# target_dist = 0.6


def affine(center, scale_self):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards

        degree = 0
        scale = 1
        crop = (0, 0)
        flip = False

        target_dist = 1
        
        img_width = 320
        img_height = 240


        A = scale * cos(degree / 180. * pi )
        B = scale * sin(degree / 180. * pi )

        scale_size = target_dist / scale_self * scale

        (width, height) = center
        center_x = width + crop[0]
        center_y = height + crop[1]

        center2zero = np.array( [[ 1., 0., -center_x],
                                 [ 0., 1., -center_y ],
                                 [ 0., 0., 1. ]] )

        rotate = np.array( [[ A, B, 0 ],
                           [ -B, A, 0 ],
                           [  0, 0, 1. ] ])

        scale = np.array( [[ scale_size, 0, 0 ],
                           [ 0, scale_size, 0 ],
                           [ 0, 0, 1. ] ])

        flip = np.array( [[ -1 if flip else 1., 0., 0. ],
                          [ 0., 1., 0. ],
                          [ 0., 0., 1. ]] )

        center2center = np.array( [[ 1., 0., img_width//2],
                                   [ 0., 1., img_height//2 ],
                                   [ 0., 0., 1. ]] )

        # order of combination is reversed
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)
        print("combined", combined, combined.shape)
        
        return combined[0:2]
    
    
# M = np.array([[ 8.57142857e-01,  0.00000000e+00, -2.60115543e+03], [ 0.00000000e+00,  8.57142857e-01, -9.97728171e+02]])
M = affine((img_width//2, img_height//2), 1)
print("M", M.shape, M)

    
img = skimage.io.imread(ID)
print("img when read", img.shape, type(img), img.dtype, img.shape)
skimage.io.imshow(img)
plt.show()

img_tx = cv2.warpAffine(img, M, (img_width-50, img_height-50), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
print("img when transformed", img_tx.shape)
skimage.io.imshow(img_tx)
plt.show()
# skimage.io.imsave(self.save_transformed_path + "/" + ID.split('/')[-1] + '_240x320_transformed.jpg', X[i, :, :, :])
print("Testing affine is done!")


# with no aug 
# combined [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
