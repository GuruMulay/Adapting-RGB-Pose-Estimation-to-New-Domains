import numpy as np
import zmq
from ast import literal_eval as make_tuple
from py_rmpe_server.py_rmpe_data_iterator import RawDataIterator

from py_rmpe_server.py_rmpe_config import RmpeCommonConfig

import six
if six.PY3:
  buffer_ = memoryview
else:
  buffer_ = buffer  # noqa


import skimage.io

save_transformed_path = None  #  '/s/red/b/nobackup/data/eggnog_cpm/eggnog_cpm_test/transformed/r8/'  # None


class DataGenCommon:
    
    def __init__(self, data_gen_eggnog, data_gen_coco):
        self.data_gen_eggnog = data_gen_eggnog
        self.data_gen_coco = data_gen_coco
        self.records = 0
        self.records_eggnog = 0
        self.records_coco = 0
      
    
    def gen_common(self):

        while True:
            yield tuple(self.gen_raw())
            
            
    def gen_raw(self):

        while True:
            if self.data_gen_eggnog:  # check if not None
                tuple_eggnog = next(self.data_gen_eggnog, None)  # None???
                assert(len(tuple_eggnog) == 2)
            if self.data_gen_coco:  # check if not None
                tuple_coco = next(self.data_gen_coco, None)  # None???
                assert(len(tuple_coco) == 2)
            
#             print("len(tuple_eggnog) len(tuple_coco)", len(tuple_eggnog), len(tuple_coco))  # len(tuple_eggnog) len(tuple_coco) 2 2# len(tuple_eggnog) len(tuple_coco) 2 2
#             print("tuple_eggnog 0 and 1 len", len(tuple_eggnog[0]), len(tuple_eggnog[1]))  # tuple_eggnog 0 and 1 2 2
#             print("tuple_eggnog 00 01 10 11", type(tuple_eggnog[0][0]), type(tuple_eggnog[0][1]), type(tuple_eggnog[1][0]), type(tuple_eggnog[1][1]))  # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
            
#             print("tuple_eggnog 00 01 10 11", tuple_eggnog[0][0].shape, tuple_eggnog[0][1].shape, tuple_eggnog[1][0].shape, tuple_eggnog[1][1].shape)  # (5, 240, 320, 3) (5, 30, 40, 11) (5, 30, 40, 11) (5, 30, 40, 11)
#             print("tuple_coco 00 01 10 11", tuple_coco[0][0].shape, tuple_coco[0][1].shape, tuple_coco[1][0].shape, tuple_coco[1][1].shape)  # (5, 368, 368, 3) (5, 46, 46, 11) (5, 46, 46, 11) (5, 46, 46, 11)

            
#             # ValueError: all the input array dimensions except for the concatenation axis must match exactly
#             x = np.concatenate((tuple_eggnog[0][0], tuple_coco[0][0]), axis=0)
#             x2 = np.concatenate((tuple_eggnog[0][1], tuple_coco[0][1]), axis=0)
            
#             y2n1 = np.concatenate((tuple_eggnog[1][0], tuple_coco[1][0]), axis=0)  # y2 stage n1
#             y2n2 = np.concatenate((tuple_eggnog[1][1], tuple_coco[1][1]), axis=0)  # y2 stage n2
            
#             # merge and shuffle the two tuples 
#             return ([x, x2], [y1, y2])

            # Alternate between these two gens
            self.records += 1
            # print("self.records =================", self.records)
        
            if self.data_gen_eggnog is None and tuple_coco is not None:
                # print("Only coco data returned")
                return tuple_coco
            
            if self.data_gen_coco is None and tuple_eggnog is not None:
                # print("Only eggnog data returned")
                return tuple_eggnog
                
            if self.records%2 == 0 and tuple_eggnog is not None:
                self.records_eggnog += 1
                # print("self.records_eggnog and tpl len", self.records_eggnog, len(tuple_eggnog), type(tuple_eggnog))  # self.records and tpl len 8 2 <class 'tuple'>
                return tuple_eggnog
            
            if self.records%2 == 1 and tuple_coco is not None:
                self.records_coco += 1
                # print("self.records_coco and tpl len", self.records_coco, len(tuple_coco), type(tuple_coco))  # self.records and tpl len 8 2 <class 'tuple'>
                return tuple_coco
            
            
class DataIteratorBase:

    def __init__(self, batch_size = 10):

        self.batch_size = batch_size

        self.split_point = 38
        self.vec_num = 38
        self.heat_num = 19

        self.keypoints = [None]*self.batch_size #this is not passed to NN, will be accessed by accuracy calculation


    def gen_raw(self): # this function used for test purposes in py_rmpe_server

        while True:
            yield tuple(self._recv_arrays())

    def gen(self, n_stages, use_eggnong_common_joints, branch_flag=0):
        batches_x, batches_x1, batches_x2, batches_y1, batches_y2 = \
            [None]*self.batch_size, [None]*self.batch_size, [None]*self.batch_size, \
            [None]*self.batch_size, [None]*self.batch_size

        sample_idx = 0

        for foo in self.gen_raw():

            if len(foo)==4:
                data_img, mask_img, label, kpts = foo
            else:
                data_img, mask_img, label = foo
                kpts = None

            # print("data_image shape", data_img.shape)  # (3, 368, 368)
            # print("mask_img shape", mask_img.shape)  # (46, 46)
            # print("label shape", label.shape)  # (57, 46, 46)
            # print("kpts shape", kpts.shape)  # (n, 18, 3)  n varies (depending on n_persons) 3 => x, y, ?
            
            # image
            dta_img = np.transpose(data_img, (1, 2, 0))  # dta_img => (368, 368, 3)
            batches_x[sample_idx]=dta_img[np.newaxis, ...]  
            # dta_img[np.newaxis, ...]  => (1, 368, 368, 3)

            # mask - the same for vec_weights, heat_weights
            vec_weights = np.repeat(mask_img[:,:,np.newaxis], self.vec_num, axis=2)
            heat_weights = np.repeat(mask_img[:,:,np.newaxis], self.heat_num, axis=2)
            
            # print("masks shapes before common: vec, heat", vec_weights.shape, heat_weights.shape)  #
            if use_eggnong_common_joints:
                vec_weights = vec_weights[:, :, RmpeCommonConfig.keep_paf_indices]
                heat_weights = heat_weights[:, :, RmpeCommonConfig.keep_joint_indices]
            # print("masks shapes after common: vec, heat", vec_weights.shape, heat_weights.shape)  #
            
            batches_x1[sample_idx]=vec_weights[np.newaxis, ...]
            batches_x2[sample_idx]=heat_weights[np.newaxis, ...]

            # label
            vec_label = label[:self.split_point, :, :]
            vec_label = np.transpose(vec_label, (1, 2, 0))
            heat_label = label[self.split_point:, :, :]
            heat_label = np.transpose(heat_label, (1, 2, 0))
            
            # print("gt shapes before common: vec, heat", vec_label.shape, heat_label.shape)  # 
            if use_eggnong_common_joints:
                vec_label = vec_label[:, :, RmpeCommonConfig.keep_paf_indices]
                # calculate background on the spot
                heat_label_no_bk = heat_label[:, :, RmpeCommonConfig.keep_joint_indices[:-1]]
                heat_label = np.dstack(( heat_label_no_bk, (1 - np.max(heat_label_no_bk[:,:,:], axis=2)) ))
                # heat_label = heat_label[:, :, RmpeCommonConfig.keep_joint_indices]
            # print("gt shapes after common: vec, heat", vec_label.shape, heat_label.shape)  #
            
            batches_y1[sample_idx]=vec_label[np.newaxis, ...]
            batches_y2[sample_idx]=heat_label[np.newaxis, ...]
        
            self.keypoints[sample_idx] = kpts
            
#             print("data_image shape", dta_img.shape, type(dta_img))  # (368, 368, 3)  <class 'numpy.ndarray'>
#             print("mask_img shape", mask_img.shape, type(mask_img))  # (46, 46)  <class 'numpy.ndarray'>
#             print("vec_label label shape", vec_label.shape, type(vec_label))  # (46, 46, 18) <class 'numpy.ndarray'>
#             print("heat_label label shape", heat_label.shape, type(heat_label))  # (46, 46, 11) <class 'numpy.ndarray'>
            
            if save_transformed_path:
                idx = np.random.randint(1000)
                skimage.io.imsave(save_transformed_path + "/" + str(idx) + '_240x320_transformed.jpg', dta_img)
                np.save(save_transformed_path + "/" + str(idx) + '_mask.npy', mask_img)
                np.save(save_transformed_path + "/" + str(idx) + '_paf30x40_transformed.npy', vec_label)
                np.save(save_transformed_path + "/" + str(idx) + '_heatmap30x40_transformed.npy', heat_label)
            
            sample_idx += 1

            if sample_idx == self.batch_size:
                sample_idx = 0

                batch_x = np.concatenate(batches_x)  # concatenate the list - batches_x
                batch_x1 = np.concatenate(batches_x1)
                batch_x2 = np.concatenate(batches_x2)
                batch_y1 = np.concatenate(batches_y1)
                batch_y2 = np.concatenate(batches_y2)

#                 print("batch_x shape", batch_x.shape)  # (2, 368, 368, 3)
#                 print("batch_x1 shape", batch_x1.shape)  # (2, 46, 46, 38)
#                 print("batch_x2 shape", batch_x2.shape)  # (2, 46, 46, 19)
#                 print("batch_y1 shape", batch_y1.shape)  # (2, 46, 46, 38)
#                 print("batch_y2 shape", batch_y2.shape)  # (2, 46, 46, 19)
                
#                 file_num = np.random.randint(0, 100)
#                 print("saving with random number ================================", file_num)
#                 np.save('./npyfiles/batch_x_' + str(file_num) + '.npy', batch_x)
#                 np.save('./npyfiles/batch_x1_' + str(file_num) + '.npy', batch_x1)
#                 np.save('./npyfiles/batch_x2_' + str(file_num) + '.npy', batch_x2)
#                 np.save('./npyfiles/batch_y1_' + str(file_num) + '.npy', batch_y1)
#                 np.save('./npyfiles/batch_y2_' + str(file_num) + '.npy', batch_y2)
                
                # x1 and x2 are the masks peculiar to COCO dataset. Not required for EGGNOG dataset
                # All y_ are ground truths. There are 12 of them because we have 6 stages x 2 outputs
#                 yield [batch_x, batch_x1, batch_x2], \
#                        [batch_y1, batch_y2,
#                         batch_y1, batch_y2,
#                         batch_y1, batch_y2,
#                         batch_y1, batch_y2,
#                         batch_y1, batch_y2,
#                         batch_y1, batch_y2]
                    
                # for n staged network
                if branch_flag == 0:
                    yield [batch_x, batch_x1, batch_x2], [batch_y1, batch_y2] * n_stages
                elif branch_flag == 1:
                    yield [batch_x, batch_x1], [batch_y1] * n_stages
                else:
                    yield [batch_x, batch_x2], [batch_y2] * n_stages

                self.keypoints = [None] * self.batch_size

    def keypoints(self):
        return self.keypoints


# class DataGeneratorClient(DataIteratorBase):

#     def __init__(self, host, port, hwm=20, batch_size=10, limit=None):

#         super(DataGeneratorClient, self).__init__(batch_size)

#         self.limit = limit
#         self.records = 0

#         """
#         :param host:
#         :param port:
#         :param hwm:, optional
#           The `ZeroMQ high-water mark (HWM)
#           <http://zguide.zeromq.org/page:all#High-Water-Marks>`_ on the
#           sending socket. Increasing this increases the buffer, which can be
#           useful if your data preprocessing times are very random.  However,
#           it will increase memory usage. There is no easy way to tell how
#           many batches will actually be queued with a particular HWM.
#           Defaults to 10. Be sure to set the corresponding HWM on the
#           receiving end as well.
#         :param batch_size:
#         :param shuffle:
#         :param seed:
#         """
#         self.host = host
#         self.port = port
#         self.hwm = hwm
#         self.socket = None

#         context = zmq.Context()
#         self.socket = context.socket(zmq.PULL)
#         self.socket.set_hwm(self.hwm)
#         self.socket.connect("tcp://{}:{}".format(self.host, self.port))


#     def _recv_arrays(self):
#         """Receive a list of NumPy arrays.
#         Parameters
#         ----------
#         socket : :class:`zmq.Socket`
#         The socket to receive the arrays on.
#         Returns
#         -------
#         list
#         A list of :class:`numpy.ndarray` objects.
#         Raises
#         ------
#         StopIteration
#         If the first JSON object received contains the key `stop`,
#         signifying that the server has finished a single epoch.
#         """

#         if self.limit is not None and self.records > self.limit:
#             raise StopIteration

#         headers = self.socket.recv_json()
#         if 'stop' in headers:
#             raise StopIteration
#         arrays = []

#         for header in headers:
#             data = self.socket.recv()
#             buf = buffer_(data)
#             array = np.frombuffer(buf, dtype=np.dtype(header['descr']))
#             array.shape = make_tuple(header['shape']) if isinstance(header['shape'], str) else header['shape']
#             # this need for comparability with C++ code, for some reasons it is string here, not tuple

#             if header['fortran_order']:
#                 array.shape = header['shape'][::-1]
#                 array = array.transpose()
#             arrays.append(array)

#         self.records += 1
#         return arrays


class DataIterator(DataIteratorBase):

    def __init__(self, file, shuffle=True, augment=True, batch_size=10, limit=None):

        super(DataIterator, self).__init__(batch_size)

        self.limit = limit
        self.records = 0

        self.raw_data_iterator = RawDataIterator(file, shuffle=shuffle, augment=augment)
        self.generator = self.raw_data_iterator.gen()


    def _recv_arrays(self):

        while True:

            if self.limit is not None and self.records > self.limit:
                raise StopIteration

            tpl = next(self.generator, None)
            if tpl is not None:
                self.records += 1
                # print("self.records and tpl len", self.records, len(tpl))  #self.records and tpl len 50 4
                return tpl

            if self.limit is None or self.records < self.limit:
                print("Staring next generator loop cycle")
                self.generator = self.raw_data_iterator.gen()
            else:
                raise StopIteration


