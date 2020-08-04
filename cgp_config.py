#!/usr/bin/env python
# -*- coding: utf-8 -*-

import concurrent.futures
import numpy as np
import cgpunet_train as cnn


# Evaluation of CNNs
def cnn_eval(args):

    dag, gpu_id, epoch_num, batchsize, batchsize_valid, dataset_path, img_format, mask_format, input_shape, target_shape, out_model, verbose = args

    print('\tgpu_id:', gpu_id, ',', dag)
    train = cnn.CNN_train(dataset_path, img_format, mask_format, verbose=verbose, input_shape=input_shape, target_shape=target_shape, batchsize=batchsize, batchsize_valid=batchsize_valid)
    try:
        evaluation = train(dag, gpu_id, epoch_num=epoch_num, out_model=out_model)
    except:
        evaluation = 0
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    return evaluation


class CNNEvaluation(object):
    def __init__(self, gpu_num, dataset_path, img_format, mask_format, verbose=True, epoch_num=50, batchsize=16, batchsize_valid=16, input_shape=(256,256,1), target_shape=(256,256,1), out_model='cgpunet'):
        self.gpu_num = gpu_num
        self.epoch_num = epoch_num
        self.batchsize = batchsize
        self.batchsize_valid = batchsize_valid
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.img_format = img_format
        self.mask_format = mask_format
        self.out_model = out_model

    def __call__(self, DAG_list, epochs, gen_num, retrain = False):
        evaluations = np.zeros(len(DAG_list))
        out_model_names = []
        for i in np.arange(0, len(DAG_list), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(DAG_list))) - i

            args = []

            for j in range(process_num):
                if not retrain:
                    out_model = self.out_model + '_' + str(gen_num) + '_' + str(j) + '.hdf5'
                else:
                    assert(isinstance(gen_num[i + j], str))
                    out_model = gen_num[i + j]
                
                out_model_names.append(out_model)
                args.append((DAG_list[i+j], j, epochs[i + j], self.batchsize, self.batchsize_valid, self.dataset_path, self.img_format, self.mask_format, self.input_shape, self.target_shape, out_model, self.verbose))

            with concurrent.futures.ProcessPoolExecutor(max_workers=self.gpu_num - 1) as executor:
                results = executor.map(cnn_eval, args)

            for k, r in enumerate(results):
                evaluations[i + k] = r
        
        return evaluations, out_model_names



# network configurations
class CgpInfoConvSet(object):
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        self.input_num = 1
        # "S_" means that the layer has a convolution layer without downsampling.
        # "D_" means that the layer has a convolution layer with downsampling.
        # "Sum" means that the layer has a skip connection.
        self.func_type = ['S_ConvBlock_1_1', 'S_ConvBlock_2_1', 'S_ConvBlock_4_1', 'S_ConvBlock_8_1', 'S_ConvBlock_16_1', 'S_ConvBlock_32_1', 'S_ConvBlock_1_3', 'S_ConvBlock_2_3', 'S_ConvBlock_4_3', 'S_ConvBlock_8_3', 'S_ConvBlock_16_3','S_ConvBlock_32_3',  
        'S_ConvBlock_1_5', 'S_ConvBlock_2_5', 'S_ConvBlock_4_5', 'S_ConvBlock_8_5', 'S_ConvBlock_16_5', 'S_ConvBlock_32_5',
                          'S_ConvBlock_128_1',    'S_ConvBlock_128_3',   'S_ConvBlock_128_5',
                          'S_ConvBlock_64_1',     'S_ConvBlock_64_3',    'S_ConvBlock_64_5',
                          'S_ConvBlock_256_1',     'S_ConvBlock_256_3',    'S_ConvBlock_256_5',
                          'S_ConvBlock_512_1',     'S_ConvBlock_512_3',    'S_ConvBlock_512_5',
                          'S_ConvBlock_1024_1',     'S_ConvBlock_1024_3',    'S_ConvBlock_1024_5',
                          'S_ResBlock_1_1',     'S_ResBlock_1_3',    'S_ResBlock_1_5',
                          'S_ResBlock_2_1',     'S_ResBlock_2_3',    'S_ResBlock_2_5',
                          'S_ResBlock_4_1',     'S_ResBlock_4_3',    'S_ResBlock_4_5',
                          'S_ResBlock_8_1',     'S_ResBlock_8_3',    'S_ResBlock_8_5',
                          'S_ResBlock_16_1',     'S_ResBlock_16_3',    'S_ResBlock_16_5',
                          'S_ResBlock_32_1',     'S_ResBlock_32_3',    'S_ResBlock_32_5',
                          'S_ResBlock_128_1',     'S_ResBlock_128_3',    'S_ResBlock_128_5',
                          'S_ResBlock_64_1',      'S_ResBlock_64_3',     'S_ResBlock_64_5',
                          'S_ResBlock_256_1',      'S_ResBlock_256_3',     'S_ResBlock_256_5',
                          'S_ResBlock_512_1',      'S_ResBlock_512_3',     'S_ResBlock_512_5',
                          'S_ResBlock_1024_1',      'S_ResBlock_1024_3',     'S_ResBlock_1024_5',
                          'D_DeconvBlock_1_1', 'D_DeconvBlock_2_1', 'D_DeconvBlock_4_1', 'D_DeconvBlock_8_1', 'D_DeconvBlock_16_1', 'D_DeconvBlock_32_1', 'D_DeconvBlock_1_3', 'D_DeconvBlock_2_3', 'D_DeconvBlock_4_3', 'D_DeconvBlock_8_3', 'D_DeconvBlock_16_3','D_DeconvBlock_32_3',
                          'D_DeconvBlock_1_5', 'D_DeconvBlock_2_5', 'D_DeconvBlock_4_5', 'D_DeconvBlock_8_5', 'D_DeconvBlock_16_5', 'D_DeconvBlock_32_5',
                          'D_DeconvBlock_128_1',    'D_DeconvBlock_128_3',   'D_DeconvBlock_128_5',
                          'D_DeconvBlock_64_1',     'D_DeconvBlock_64_3',    'D_DeconvBlock_64_5',
                          'D_DeconvBlock_256_1',     'D_DeconvBlock_256_3',    'D_DeconvBlock_256_5',
                          'D_DeconvBlock_512_1',     'D_DeconvBlock_512_3',    'D_DeconvBlock_512_5',
                          'D_DeconvBlock_1024_1',     'D_DeconvBlock_1024_3',    'D_DeconvBlock_1024_5',
                          'Concat', 'Sum', 'Concat', 'Sum', 'Concat', 'Sum', 'Concat', 'Sum', 'Concat', 'Sum', 'Concat', 'Sum', 'Concat', 'Sum', 'Concat', 'Sum',
                          'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool', 'Max_Pool', 'Avg_Pool']
                          
        self.func_in_num = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1,     1,    1,
                          1,  1,     1,
                          1,  1,     1,
                          1,  1,     1,
                          1,  1,     1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1,  1,     1,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        self.out_num = 1

        #Tabish 04/06/2020: I think this should be changed to same as func_type
        #self.out_type = ['full']
        self.out_type = self.func_type
        #self.out_in_num = [1]
        self.out_in_num = self.func_in_num

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])

    
    def get_func_input_num(self, func_num):
        try:
            in_num = self.func_in_num[func_num]
        except:
            raise KeyError

        return in_num
