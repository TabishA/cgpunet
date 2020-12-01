import argparse
from datetime import datetime
from cgp import *
from cgp_config import *
from dataHelperTif import *
from params import *
from nsga import *

if __name__ == "__main__":
    network_info = CgpInfoConvSet(rows=5, cols=30, level_back=30, min_active_num=1, max_active_num=30)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-path', type=dir_path)
    #
    # args = parser.parse_args()
    # data_path = os.path.dirname(args.path)
    data_path = DATA_PATH
    print('data path: {}'.format(data_path))

    eval_f = CNNEvaluation(gpu_num=4, dataset_path=data_path, img_format='*.tif', mask_format='*.tif',
                           input_shape=(128, 128, 1), target_shape=(128, 128, 1))
    # CGP object contains a population of pop_size individuals
    cgp = CGP(network_info, eval_f, max_eval=100, pop_size=20, lam=4, imgSize=128, init=False)

    population = get_files(LOAD_POP_PATH, '*_0.p')

    # set cgp.pop to the loaded population and reset initial generation number to 0
    cgp.load_population(population[0], init_gen=0)
    print("The loaded population contains {} individuals".format(len(cgp.pop)))

    # Sample solutions on pareto front
    cgp.pop = identify_pareto(cgp.pop)
    print("{} individuals were sampled".format(len(cgp.pop)))

    # If resuming annealing process, load the state
    # p_file = get_files('./sa_files_netlists', 'annealing_*.p')
    # cgp.simulated_annealing(mutation_rate=0.1, log_dir="SA_Batch1_",
    #                             timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), load=True, load_file=p_file)

    # Perform annealing for each solution and optimize it
    cgp.simulated_annealing(mutation_rate=0.1, log_dir="SA_Batch1_",
                            timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))