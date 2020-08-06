#cgpunet_main.py

import argparse
from cgp import *
from cgp_config import *
import os
import sys



def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


if __name__ == "__main__":
    #from cgp_config, initialize CGP grid representation
    #level_back = cols
    network_info = CgpInfoConvSet(rows=5, cols=30, level_back=30, min_active_num=1, max_active_num=30)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=dir_path)
    
    args = parser.parse_args()
    dir_name = os.path.dirname(args.path)
    
    print('data path: {}'.format(dir_name))
    eval_f = CNNEvaluation(gpu_num=4, dataset_path=dir_name, img_format='*.tif', mask_format='*.tif', input_shape=(128,128,1), target_shape=(128,128,1), out_model='cgpunet_drive')
    
    #CGP object contains a population of pop_size individuals
    cgp = CGP(network_info, eval_f, max_eval=100, pop_size=100, lam=4, imgSize=128, init=False)
    cgp.modified_evolution(mutation_rate=0.1)
