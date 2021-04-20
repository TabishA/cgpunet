#cgpunet_main.py

import argparse
from cgp import *
from cgp_config import *
from dataHelperTif import *
import os
import sys
from tensorflow.keras.utils import to_categorical



def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


if __name__ == "__main__":
    #from cgp_config, initialize CGP grid representation
    #level_back = cols
    network_info = CgpInfoConvSet(rows=5, cols=30, level_back=30, min_active_num=1, max_active_num=30)
    
    #loading datasets
    x_train = pickle.load(open('x_train.p', 'rb'))
    y_train = pickle.load(open('y_train.p', 'rb'))
    x_valid = pickle.load(open('x_valid.p', 'rb'))
    y_valid = pickle.load(open('y_valid.p', 'rb'))

    x_train = x_train/255
    x_valid = x_valid/255
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    eval_f = CNNEvaluation(4, x_train, y_train, x_valid, y_valid, input_shape=(28,28,1), batchsize=32, batchsize_valid=32)
    
    #CGP object contains a population of pop_size individuals
    cgp = CGP(network_info, eval_f, max_eval=100, pop_size=20, lam=4, imgSize=28, init=False, basename='cgp_mnist')
    
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


    #run the EA
    cgp.modified_evolution(mutation_rate=0.1)
