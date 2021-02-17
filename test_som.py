from som_cosine2 import *
from cgp import *
from cgp_config import *
from cgp_2_dag import *
from dag_2_cnn import *
from dataHelperTif import *

#from cgp_config, initialize CGP grid representation
#level_back = cols
network_info = CgpInfoConvSet(rows=5, cols=30, level_back=30, min_active_num=1, max_active_num=30)

#NOTE: change gpu_num to 4 when on compute canada
#(self, gpu_num, dataset_path, img_format, mask_format, verbose=True, epoch_num=50, batchsize=16, batchsize_valid=1, input_shape=(256,256,1), target_shape=(256,256,1), out_model='cgpunet')

dp = 'C:\\Users\\tabis\\Documents\\NSERC_USRA\\datasets\\DRIVE'

eval_f = CNNEvaluation(gpu_num=1, dataset_path=dp, img_format='*.tif', mask_format='*.tif', input_shape=(96,96,1), target_shape=(96,96,1))

pop = [Individual(network_info, False) for _ in range(100)]

test_som = SOM(pop, network_info, map_size=(10,10), input_size=(96,96))

all_populations = get_files('./p_files_netlists/', '*.p')
all_populations.sort()

for i, p in enumerate(all_populations):
    loaded_pop = pickle.load(open(p, 'rb'))
    test_som.fit(population=loaded_pop, t_max=100)
    test_som.draw(save_dir = './som_cosine_figures/{}.png'.format(i))
