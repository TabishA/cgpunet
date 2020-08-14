#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cgpunet_train as cnn
import multiprocessing

# Evaluation of CNNs
def cnn_eval(dag, gpu_id, epoch_num, batchsize, batchsize_valid, dataset_path, img_format, mask_format, input_shape, target_shape, out_model, verbose, return_dict):

    #dag, gpu_id, epoch_num, batchsize, batchsize_valid, dataset_path, img_format, mask_format, input_shape, target_shape, out_model, verbose, return_dict = args

    print('\tgpu_id:', gpu_id, ',', dag)
    train = cnn.CNN_train(dataset_path, img_format, mask_format, verbose=verbose, input_shape=input_shape, target_shape=target_shape, batchsize=batchsize, batchsize_valid=batchsize_valid)
    try:
        evaluation = train(dag, gpu_id, epoch_num=epoch_num, out_model=out_model)
    except Exception as e:
        print(e)
        evaluation = 0
    print('\tgpu_id:', gpu_id, 'model_name: ', out_model, ', eval:', evaluation)
    return_dict[out_model] = evaluation



class CNNEvaluation(object):
    def __init__(self, gpu_num, dataset_path, img_format, mask_format, verbose=True, epoch_num=50, batchsize=16, batchsize_valid=16, input_shape=(256,256,1), target_shape=(256,256,1)):
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

    def __call__(self, DAG_list, epochs, model_names, retrain = False):
        manager = multiprocessing.Manager()
        return_dict = manager.dict() 
        for i in np.arange(0, len(DAG_list), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(DAG_list))) - i

            processes = []
            arguments = []      

            for j in range(process_num):
                out_model = model_names[i + j]
                arguments = (DAG_list[i+j], j, epochs[i + j], self.batchsize, self.batchsize_valid, self.dataset_path, self.img_format, self.mask_format, self.input_shape, self.target_shape, out_model, self.verbose, return_dict)
                p = multiprocessing.Process(target=cnn_eval, args=arguments)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        
        #the current version is throwing key errors with some model names.
        #here, I am ensuring that the return dict has values for corresponding to all elements of model_names by repeating the above process for missing models
        #TODO: remove repeated code by incorporating into the loop above
        missed_models = []
        
        for name in model_names:
            if name not in return_dict.keys(): missed_models.append(name)

        
        for i in np.arange(0, len(missed_models), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(missed_models))) - i
            
            processes = []
            arguments = []      
            
            for j in range(process_num):
                out_model = missed_models[i + j]
                k = model_names.index(out_model)
                arguments = (DAG_list[k], j, epochs[k], self.batchsize, self.batchsize_valid, self.dataset_path, self.img_format, self.mask_format, self.input_shape, self.target_shape, out_model, self.verbose, return_dict)
                p = multiprocessing.Process(target=cnn_eval, args=arguments)
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        
        assert(len(return_dict) == len(model_names))

        return return_dict



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
