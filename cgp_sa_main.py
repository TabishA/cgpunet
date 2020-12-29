import argparse
from datetime import datetime
from cgp import *
from cgp_config import *
from dataHelperTif import *
from params import *
from nsga import *

if __name__ == "__main__":

    # Cluster Run
    start_run = datetime.now()
    network_info = CgpInfoConvSet(rows=5, cols=30, level_back=30, min_active_num=1, max_active_num=30)
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=dir_path)
    args = parser.parse_args()
    data_path = os.path.dirname(args.path)
    print('data path: {}'.format(data_path))
    eval_f = CNNEvaluation(gpu_num=4, dataset_path=data_path, img_format='*.tif', mask_format='*.tif',
                           input_shape=(128, 128, 1), target_shape=(128, 128, 1))
    cgp = CGP(network_info, eval_f, max_eval=100, pop_size=20, lam=4, imgSize=128, init=False, basename='sa')

    # New Run
    cgp.load_population(LOAD_POP_PATH, init_gen=0)
    print("The loaded population contains {} individuals".format(len(cgp.pop)))
    cgp.pop = identify_pareto(cgp.pop)
    print("{} individuals were sampled".format(len(cgp.pop)))

    cgp.simulated_annealing(mutation_rate=0.1,
                            start_temp=25,
                            final_temp=0.01,
                            alpha=0.05,
                            cooling='linear',
                            consider_neutral=True,
                            log_dir="SA_Neu_",
                            timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Resuming another run
    # cgp.simulated_annealing(load=True, load_file='annealing_23.049999999999972.p')

    print('Time Elapsed is {}'.format(datetime.now() - start_run))




