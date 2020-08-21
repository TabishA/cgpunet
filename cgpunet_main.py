#cgpunet_main.py

import argparse
from cgp import *
from cgp_config import *
from dataHelperTif import *
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
    
    #dir_name = 'C:\\Users\\tabis\\Documents\\NSERC_USRA\\datasets\\DRIVE'
    
    print('data path: {}'.format(dir_name))
    eval_f = CNNEvaluation(gpu_num=4, dataset_path=dir_name, img_format='*.tif', mask_format='*.tif', input_shape=(128,128,1), target_shape=(128,128,1))
    
    #CGP object contains a population of pop_size individuals
    cgp = CGP(network_info, eval_f, max_eval=100, pop_size=100, lam=4, imgSize=128, init=False)
    
    #mode='novelty' runs the Novelty Search and saves each generation as ./p_files_netlists/population_novelty_{num_gen}.p
    cgp.modified_evolution(mutation_rate=0.1, mode='novelty')

    #get the final population which is optimized for novelty
    novel_populations = get_files('./p_files_netlists/', '*_novelty*.p')
    novel_populations.sort()
    init_novel_population = novel_populations[len(novel_populations) - 1]

    #set cgp.pop to the loaded population and reset initial generation number to 0
    cgp.load_population(init_novel_population, init_gen=0)

    #run the EA
    cgp.modified_evolution(mutation_rate=0.1)
