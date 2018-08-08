from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
import numpy as np

# dropout
from keras.layers import SpatialDropout2D

from py_eggnog_server.py_eggnog_config import EggnogGlobalConfig

# stages = 6  # removed for only 1 staged model testing on eggnog
# original cpm with COCO
np_branch1 = 18  # 18 (keeping only common joints and paf pairs) # 38
np_branch2 = 11  # 11 (keeping only common joints and paf pairs) # 19

# modified cpm with EGGNOG
# DOES NOT work here because the n_hm and n_paf are not updated as per update function in train_*.py main file
# np_branch1 = EggnogGlobalConfig.n_paf
# np_branch2 = EggnogGlobalConfig.n_hm

# to test out incremental or decremental dropout
decreasing_dropout = True
spatial_dropout_rates_stage_1 = [0.25, 0.20, 0.15, 0.10, 0.05]
spatial_dropout_rates_stage_t = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10, 0.05]  # symmetric in ascending and descending order

if not decreasing_dropout:
    spatial_dropout_rates_stage_1 = spatial_dropout_rates_stage_1[::-1]
    spatial_dropout_rates_stage_t = spatial_dropout_rates_stage_t[::-1]

print("spatial_dropout_rates_stage_1, spatial_dropout_rates_stage_t", spatial_dropout_rates_stage_1, spatial_dropout_rates_stage_t)


def relu(x): return Activation('relu')(x)

def conv(x, nf, ks, name, weight_decay, spatial_dropout_rate):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    
    # added spatial dropout (7/11)
    # print("Added spatial dropout")
    x = SpatialDropout2D(rate=spatial_dropout_rate, name=name+"_sdropout", data_format='channels_last')(x)

    return x


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def vgg_block(x, weight_decay):
    vgg_dropout_rate = 0.2
    print("vgg_dropout_rate", vgg_dropout_rate)
    
    # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0), vgg_dropout_rate)  # Assign_22
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0), vgg_dropout_rate)
    x = relu(x)

    return x


# def stage1_block(x, num_p, branch, weight_decay):  # names from gh repo
#     # Block 1
#     x = conv(x, 128, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))  # Assign_40
#     x = relu(x)
#     x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))  # _36
#     x = relu(x)
#     x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))  # _32
#     x = relu(x)
#     x = conv(x, 512, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))  # _28
#     x = relu(x)
#     x = conv(x, num_p, 1, "Mconv5_stage1_L%d_EGGNOG" % branch, (weight_decay, 0))  # _26

#     return x


# Using this for warm-starting prototype. The following names match with model.h5 file's names.
def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch, (weight_decay, 0), spatial_dropout_rates_stage_1[0])  # Assign_40
    x = relu(x)
    x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch, (weight_decay, 0), spatial_dropout_rates_stage_1[1])  # _36
    x = relu(x)
    x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch, (weight_decay, 0), spatial_dropout_rates_stage_1[2])  # _32
    x = relu(x)
    x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch, (weight_decay, 0), spatial_dropout_rates_stage_1[3])  # _28
    x = relu(x)
    x = conv(x, num_p, 1, "conv5_5_CPM_L%d_EGGNOG" % branch, (weight_decay, 0), spatial_dropout_rates_stage_1[4])  # _26
    
#     # added spatial dropout (7/10)
#     print("Added spatial dropout")
#     x = SpatialDropout2D(rate=0.5, data_format='channels_last')(x)
    
    return x



"""
Renamed 1st conv layer in following method (for stage 2 onwards) because original paper has 128 (from VGG) + 38 (pafs) + 19 (hms) = 185 chanelled image (46x46) at the end of stage 1 or even stage n. So the saved weights for this conv layer has a shape of [128, 185, 7, 7] when loaded using load_weights(by_name=True). With eggnog dataset n_channels at the end of stage 1 or stage n is 128 (from VGG) + 36 (pafs) + 20 (hms) = 184 or less depending on the branch_flag [branch_flag = 0  # 0 => both branches; 1 => branch L1 only; 2 => branch L2 only (heatmaps only)]. Therefore, the original weights from model.h5 for 1st conv layer at each stage (>1) cannot be loaded into this modified eggnog weights which are shaped (184, 128, 7, 7). So, we need to rename these layers to avoid loading the weights by name.
"""

def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 128, 7, "Mconv1_stage%d_L%d_EGGNOG" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[0])  # _24
    x = relu(x)
    x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[1])
    x = relu(x)
    x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[2])
    x = relu(x)
    x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[3])
    x = relu(x)
    x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[4])
    x = relu(x)
    x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[5])
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d_EGGNOG" % (stage, branch), (weight_decay, 0), spatial_dropout_rates_stage_t[6])

#     # added spatial dropout (7/10)
#     print("Added spatial dropout")
#     x = SpatialDropout2D(rate=0.5, data_format='channels_last')(x)
    
    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == np_branch1:
        w = Multiply(name=w_name)([x, mask1])  # vec_weight
    elif num_p == np_branch2:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    else:
        assert False, "wrong number of layers num_p=%d " % num_p
    return w


def get_training_model_eggnog_v1(weight_decay, gpus=None, stages=6, branch_flag=0):
    """
    This model has an additional flag for heatmap only network architecture 
    """
#     assert(np_branch2 == EggnogGlobalConfig.n_hm)
#     assert(np_branch1 == EggnogGlobalConfig.n_paf)
    
    np_branch1 = EggnogGlobalConfig.n_paf
    np_branch2 = EggnogGlobalConfig.n_hm
    
    print("np_branch1, np_branch2", np_branch1, np_branch2)
#     print("Updated joint info:")
#     print("Final EggnogGlobalConfig.joint_indices", EggnogGlobalConfig.joint_indices)
#     print("Final EggnogGlobalConfig.paf_pairs_indices", EggnogGlobalConfig.paf_pairs_indices)
#     print("Final EggnogGlobalConfig.paf_indices", EggnogGlobalConfig.paf_indices_xy)
#     print("EggnogGlobalConfig.n_kp, n_hm, n_paf", EggnogGlobalConfig.n_kp, EggnogGlobalConfig.n_hm, EggnogGlobalConfig.n_paf)

    img_input_shape = (None, None, 3)
    # to print the shapes at the output of every layer
    # img_input_shape = (240, 320, 3)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    inputs.append(img_input)  # unncessary, but used to append inputs and feed them to Model() class down below

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    if branch_flag == 2:  # heatmaps only
        # stage 1 - branch 2 (confidence maps)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)

        x = Concatenate()([stage1_branch2_out, stage0_out])
        
        outputs.append(stage1_branch2_out)
        
        # stage sn >= 2
        for sn in range(2, stages + 1):
            # stage SN - branch 2 (confidence maps)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)

            outputs.append(stageT_branch2_out)

            if (sn < stages):
                x = Concatenate()([stageT_branch2_out, stage0_out])
                
    elif branch_flag == 0:  # both heatmap and pafs   
        # stage 1 - branch 1 (PAF)
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)

        # stage 1 - branch 2 (confidence maps)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)

        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        outputs.append(stage1_branch1_out)
        outputs.append(stage1_branch2_out)

        # stage sn >= 2
        for sn in range(2, stages + 1):
            # stage SN - branch 1 (PAF)
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
            # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
            # w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

            # stage SN - branch 2 (confidence maps)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
            # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
            # w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

            # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
            # outputs.append(w1)
            # outputs.append(w2)

            outputs.append(stageT_branch1_out)
            outputs.append(stageT_branch2_out)

            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    else:
        raise NotImplementedError("Only paf network is not implemented")
        
    if gpus is None:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        import tensorflow as tf
        with tf.device('/cpu:0'): #this model will not be actually used, it's template
            model = Model(inputs=inputs, outputs=outputs)

    return model


def get_training_model_eggnog(weight_decay, gpus=None, stages=6):
    
    img_input_shape = (None, None, 3)
    # to print the shapes at the output of every layer
    # img_input_shape = (240, 320, 3)
    
    # don't have these masks in EGGNOG
    # vec_input_shape = (None, None, 36)  # 36 pafs 
    # heat_input_shape = (None, None, 20)  # 20 heatmaps

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    # don't have these masks in EGGNOG
    # vec_weight_input = Input(shape=vec_input_shape)
    # heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    # don't have these masks in EGGNOG
    # inputs.append(vec_weight_input)
    # inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
    # w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
    # w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # don't have to apply masks becuase eggnog has everython labeled and only one person per frame
    # outputs.append(w1)
    # outputs.append(w2)
    
    outputs.append(stage1_branch1_out)
    outputs.append(stage1_branch2_out)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
        # w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
        # w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        # don't have to apply masks becuase eggnog has everything labeled and only one person per frame
        # outputs.append(w1)
        # outputs.append(w2)
        
        outputs.append(stageT_branch1_out)
        outputs.append(stageT_branch2_out)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    if gpus is None:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        import tensorflow as tf
        with tf.device('/cpu:0'): #this model will not be actually used, it's template
            model = Model(inputs=inputs, outputs=outputs)

    return model


def get_training_model(weight_decay, gpus=None, stages=6):

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, np_branch1)
    heat_input_shape = (None, None, np_branch2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    if gpus is None:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        import tensorflow as tf
        with tf.device('/cpu:0'):  # this model will not be actually used, it's template
            model = Model(inputs=inputs, outputs=outputs)

    return model


def get_training_model_common(weight_decay, gpus=None, stages=6, branch_flag=0):

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, np_branch1)
    heat_input_shape = (None, None, np_branch2)
    
    print("Verifying if EGGNOG and COCO specified joints and paf count is the same.")
#     assert(np_branch1 == EggnogGlobalConfig.n_paf)  # EGGNOG has three less pafs as compared to COCO
    assert(np_branch2 == EggnogGlobalConfig.n_hm)
    
    print("EggnogGlobalConfig.n_kp, n_hm, n_paf", EggnogGlobalConfig.n_kp, EggnogGlobalConfig.n_hm, EggnogGlobalConfig.n_paf)
    print("np_branch1, np_branch2", np_branch1, np_branch2)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    if branch_flag == 2:  # heatmaps only
        inputs.append(img_input)
        inputs.append(heat_weight_input)
        
        # stage 1 - branch 2 (confidence maps)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
        w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)
        
        x = Concatenate()([stage1_branch2_out, stage0_out])
        
        outputs.append(w2)
        
        # stage sn >= 2
        for sn in range(2, stages + 1):
            # stage SN - branch 2 (confidence maps)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
            w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)
            
            outputs.append(w2)

            if (sn < stages):
                x = Concatenate()([stageT_branch2_out, stage0_out])
                
    elif branch_flag == 0:  # both heatmap and pafs  
        inputs.append(img_input)
        inputs.append(vec_weight_input)
        inputs.append(heat_weight_input)
    
        # stage 1 - branch 1 (PAF)
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
        w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)
        
        # stage 1 - branch 2 (confidence maps)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
        w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)
        
        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        outputs.append(w1)
        outputs.append(w2)
        
        
        # stage sn >= 2
        for sn in range(2, stages + 1):
            # stage SN - branch 1 (PAF)
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
            w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

            # stage SN - branch 2 (confidence maps)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
            w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

            outputs.append(w1)
            outputs.append(w2)

            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    else:
        raise NotImplementedError("Only paf network is not implemented")
        
        
    if gpus is None:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        import tensorflow as tf
        with tf.device('/cpu:0'):  # this model will not be actually used, it's template
            model = Model(inputs=inputs, outputs=outputs)

    return model



# # this one is for COCO dataset
# def get_training_model(weight_decay, gpus=None):

#     img_input_shape = (None, None, 3)
#     vec_input_shape = (None, None, 38)
#     heat_input_shape = (None, None, 19)

#     inputs = []
#     outputs = []

#     img_input = Input(shape=img_input_shape)
#     vec_weight_input = Input(shape=vec_input_shape)
#     heat_weight_input = Input(shape=heat_input_shape)

#     inputs.append(img_input)
#     inputs.append(vec_weight_input)
#     inputs.append(heat_weight_input)

#     img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

#     # VGG
#     stage0_out = vgg_block(img_normalized, weight_decay)

#     # stage 1 - branch 1 (PAF)
#     stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
#     w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1)

#     # stage 1 - branch 2 (confidence maps)
#     stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
#     w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2)

#     x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

#     outputs.append(w1)
#     outputs.append(w2)

#     # stage sn >= 2
#     for sn in range(2, stages + 1):
#         # stage SN - branch 1 (PAF)
#         stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
#         w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1)

#         # stage SN - branch 2 (confidence maps)
#         stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
#         w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2)

#         outputs.append(w1)
#         outputs.append(w2)

#         if (sn < stages):
#             x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

#     if gpus is None:
#         model = Model(inputs=inputs, outputs=outputs)
#     else:
#         import tensorflow as tf
#         with tf.device('/cpu:0'):  # this model will not be actually used, it's template
#             model = Model(inputs=inputs, outputs=outputs)

#     return model


def get_testing_model_eggnog_v1(stages=2, branch_flag=2):

    img_input_shape = (None, None, 3)
    inputs = []
    outputs = []
    
    img_input = Input(shape=img_input_shape)
    inputs.append(img_input)
    
    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)
#     print("##### img_normlaized ", np.max(img_normalized ), np.min(img_normalized))
    
    # VGG
    stage0_out = vgg_block(img_normalized, None)

    
    if branch_flag == 0:  # heatmaps and pafs both
        # stage 1
        stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)
        x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
            if (sn < stages):
                x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
        
    elif branch_flag == 2:  # heatmaps only
        # stage 1
        stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)
        x = Concatenate()([stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
            if (sn < stages):
                x = Concatenate()([stageT_branch2_out, stage0_out])

        model = Model(img_input, [stageT_branch2_out])

    else:
        raise NotImplementedError("Only-paf network is not implemented")
    

    return model


# def get_testing_model_eggnog(stages=6):

#     img_input_shape = (None, None, 3)
#     inputs = []
#     outputs = []
    
#     img_input = Input(shape=img_input_shape)
#     inputs.append(img_input)
#     print(img_input)
#     img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]
#     print("##### img_normlaized ", np.max(img_normalized ), np.min(img_normalized ))
#     # VGG
#     stage0_out = vgg_block(img_normalized, None)

#     # stage 1 - branch 1 (PAF)
#     stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

#     # stage 1 - branch 2 (confidence maps)
#     stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

#     x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])
    
#     outputs.append(stage1_branch1_out)
#     outputs.append(stage1_branch2_out)
    
#     # stage t >= 2
#     stageT_branch1_out = None
#     stageT_branch2_out = None
#     for sn in range(2, stages + 1):
#         stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
#         stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
        
#         outputs.append(stageT_branch1_out)
#         outputs.append(stageT_branch2_out)
        
#         if (sn < stages):
#             x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

#     model = Model(inputs=inputs, outputs=outputs)

#     return model


def get_testing_model(stages=6):

    img_input_shape = (None, None, 3)
    
    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model


