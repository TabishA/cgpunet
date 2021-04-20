from cgp import *
from knn import *
from cgp_config import *
import argparse

class NoveltySearch(object):
    def __init__(self, cgp):
        self.cgp = cgp
        self.archive = []
        self.DAG_list = []
        for p in cgp.pop:
            G = cgp_2_dag(p.active_net_list())
            DAG_list.append(G)

    def __call__(self):
        pass



if __name__ == "__main__":
    network_info = CgpInfoConvSet(rows=5, cols=30, level_back=30, min_active_num=1, max_active_num=30)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', type=dir_path)
    
    args = parser.parse_args()
    dir_name = os.path.dirname(args.path)
    
    print('data path: {}'.format(dir_name))
    eval_f = CNNEvaluation(gpu_num=4, dataset_path=dir_name, img_format='*.tif', mask_format='*.tif', input_shape=(128,128,1), target_shape=(128,128,1), out_model='cgpunet_drive')
    
    #CGP object contains a population of pop_size individuals
    cgp = CGP(network_info, eval_f, max_eval=100, pop_size=100, lam=4, imgSize=128, init=False)

    #DAG_list = []
    #for p in cgp.pop: DAG_list.append(cgp_2_dag(p.active_net_list()))
