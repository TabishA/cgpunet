#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import time
import numpy as np
import math
import networkx as nx
from cgp_2_dag import *
from dag_2_cnn import *
import matplotlib.pyplot as plt
import pickle
from knn import *
from tensorflow.compat.v1.keras import backend as K
import multiprocessing


#https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
def get_model_memory_usage(net_list, batch_size, input_shape, target_shape, return_dict):
    try:
        model = dag_2_cnn(cgp_2_dag(net_list), 0, input_shape, target_shape, compile=False)
    except KeyError as e:
        print(e)
        return 1000
    except:
        print(e)
        raise

    shapes_mem_count = 0
    internal_model_mem_count = 0

    for l in model.layers:
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
            
        shapes_mem_count += single_layer_mem
        
    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    number_size = 4.0

    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0
        
    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    K.clear_session()
    
    return_dict["memory"] = gbytes

# gene[f][c] f:function type, c:connection (nodeID)
class Individual(object):

    def __init__(self, net_info, init):
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.is_pool = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = 0
        self.epochs_trained = 0
        self.trainable_params = 0
        self.gen_num = 0
        self.model_name = ''
        self.novelty = 0
        if init:
            print('init with specific architectures')
            self.init_gene_with_conv() # In the case of starting only convolution
        else:
            self.init_gene()           # generate initial individual randomly

    def init_gene_with_conv(self):
        # initial architecture
        arch = ['S_ConvBlock_64_3']
       
        input_layer_num = int(self.net_info.input_num / self.net_info.rows) + 1
        output_layer_num = int(self.net_info.out_num / self.net_info.rows) + 1
        layer_ids = [((self.net_info.cols - 1 - input_layer_num - output_layer_num) + i) // (len(arch)) for i in range(len(arch))]
        prev_id = 0 # i.e. input layer
        current_layer = input_layer_num
        block_ids = []  # *do not connect with these ids
        
        # building convolution net
        for i, idx in enumerate(layer_ids):
            
            current_layer += idx
            n = current_layer * self.net_info.rows + np.random.randint(self.net_info.rows)
            block_ids.append(n)
            self.gene[n][0] = self.net_info.func_type.index(arch[i])
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            
            self.gene[n][1] = prev_id
            for j in range(1, self.net_info.max_in_num):
                self.gene[n][j + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)
            
            prev_id = n + self.net_info.input_num
        
        # output layer        
        n = self.net_info.node_num
        type_num = self.net_info.func_type_num
        self.gene[n][0] = np.random.randint(type_num)
        col = np.min((int(n / self.net_info.rows), self.net_info.cols))
        max_connect_id = col * self.net_info.rows + self.net_info.input_num
        min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
            if col - self.net_info.level_back >= 0 else 0
        
        self.gene[n][1] = prev_id
        for i in range(1, self.net_info.max_in_num):
            self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)        
        block_ids.append(n) 
           
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            
            if n in block_ids:
                continue
            
            # type gene
            type_num = self.net_info.func_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

        self.check_active()

    def init_gene(self):
        # intermediate node
        for n in range(self.net_info.node_num + self.net_info.out_num):
            # type gene
            type_num = self.net_info.func_type_num
            self.gene[n][0] = np.random.randint(type_num)
            # connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            
            # we do not allow 2 input function nodes at index 1, since this will only merge the input with itself
            while self.net_info.get_func_input_num(self.gene[n][0]) == 2:
                if max_connect_id == 1:
                    self.gene[n][0] = np.random.randint(type_num)
                else:
                    break
            
            for i in range(self.net_info.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

            while self.gene[n][1] == self.gene[n][2] and self.net_info.get_func_input_num(self.gene[n][0]) == 2:
                self.gene[n][2] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)
                #print('max: {}, min: {}, gene: {}'.format(max_connect_id, min_connect_id, self.gene[n][2]))

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i+1] >= self.net_info.input_num:
                    self.__check_course_to_out(self.gene[n][i+1] - self.net_info.input_num)

    def check_active(self):
        # clear
        self.is_active[:] = False
        # start from output nodes
        for n in range(self.net_info.out_num):
            self.__check_course_to_out(self.net_info.node_num + n)
    
    def check_pool(self):
        G = cgp_2_dag(self.active_net_list())
        max_pool_num = 0
        for n in G.nodes():
            pf = G.nodes[n]['pool_factor']
            if pf > max_pool_num:
                max_pool_num = pf
        
        return max_pool_num

    def __mutate(self, current, min_int, max_int):
        mutated_gene = current
        while current == mutated_gene:
            mutated_gene = min_int + np.random.randint(max_int - min_int)
        return mutated_gene

    def mutation(self, mutation_rate=0.01):
        active_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                if self.is_active[n]:
                    active_check = True
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if np.random.rand() < mutation_rate and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)
                    if self.is_active[n] and i < in_num:
                        active_check = True

        self.check_active()
        return active_check

    def neutral_mutation(self, mutation_rate=0.01):
        print('NEUTRAL MUTATION - Before: {}'.format(self.active_net_list()))
        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if not self.is_active[n] and np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
            # mutation for connection gene
            col = np.min((int(n / self.net_info.rows), self.net_info.cols))
            max_connect_id = col * self.net_info.rows + self.net_info.input_num
            min_connect_id = (col - self.net_info.level_back) * self.net_info.rows + self.net_info.input_num \
                if col - self.net_info.level_back >= 0 else 0
            in_num = self.net_info.func_in_num[t] if n < self.net_info.node_num else self.net_info.out_in_num[t]
            for i in range(self.net_info.max_in_num):
                if (not self.is_active[n] or i >= in_num) and np.random.rand() < mutation_rate \
                        and max_connect_id - min_connect_id > 1:
                    self.gene[n][i+1] = self.__mutate(self.gene[n][i+1], min_connect_id, max_connect_id)

        self.check_active()
        print('After: {}'.format(self.active_net_list()))
        return False

    def count_active_node(self):
        return self.is_active.sum()

    def copy(self, source):
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.is_active = source.is_active.copy()
        self.eval = source.eval

    def active_net_list(self):
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:    # intermediate node
                    type_str = self.net_info.func_type[t]
                else:    # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i+1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)
        
        return net_list
    

# CGP with (1 + \lambda)-ES
class CGP(object):
    def __init__(self, net_info, eval_func, max_eval, pop_size=100, lam=4, gpu_mem=16, imgSize=256, init=False, basename = 'cgpunet_drive'):
        self.lam = lam
        #GPU memory in GB
        self.gpu_mem = gpu_mem
        self.net_info = net_info
        self.pop_size = pop_size
        self.pop = [Individual(self.net_info, init) for _ in range(self.pop_size)]
        self.eval_func = eval_func
        self.num_gen = 0
        self.num_eval = 0
        self.max_pool_num = int(math.log2(imgSize) - 2)
        self.max_eval = max_eval
        self.init = init
        self.fittest = None
        self.basename = basename
        self.search_archive = []
        self.epsilon = 0.05
    

    def pickle_population(self, save_dir, mode='eval'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        p_name = os.path.join(save_dir, 'population_' + mode + '_' + str(self.num_gen) + '.p')

        try:
            pickle.dump(self.pop, open(p_name, "wb"))
        except:
            pass

    
    def load_population(self, p_file, init_gen):
        loaded_pop = pickle.load(open(p_file, "rb"))
        self.pop = loaded_pop
        self.num_gen = init_gen


    def _evaluation(self, pop, eval_flag, init_flag=False):
        # create network list
        DAG_list = []
        model_names = []
        active_index = np.where(eval_flag)[0]
        for i in active_index:
            G = cgp_2_dag(pop[i].active_net_list())
            DAG_list.append(G)
            model_names.append(pop[i].model_name)

        num_epochs = self.num_epochs_scheduler(self.num_eval, self.max_eval, self.eval_func.epoch_num)
        num_epochs_list = [num_epochs]*len(DAG_list)
        
        # evaluation
        fp = self.eval_func(DAG_list, num_epochs_list, model_names)
        for i, j in enumerate(active_index):
            pop[j].gen_num = self.num_gen
            pop[j].eval = fp[pop[j].model_name][0]
            pop[j].trainable_params = fp[pop[j].model_name][1]
            pop[j].epochs_trained = num_epochs
        
        evaluations = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations[i] = pop[i].eval

        if not init_flag:
            self.num_eval += len(DAG_list)
        return evaluations

    def _log_data(self, net_info_type='active_only', start_time=0):
        log_list = [self.num_gen, self.num_eval, time.time()-start_time, self.pop[0].eval, self.pop[0].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[0].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[0].gene.flatten().tolist()
        else:
            pass
        return log_list

    def _log_data_children(self, net_info_type='active_only', start_time=0, pop=None):
        log_list = [self.num_gen, self.num_eval, time.time()-start_time, pop.eval, pop.count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        else:
            pass
        return log_list

    def load_log(self, log_data):
        self.num_gen = log_data[0]
        self.num_eval = log_data[1]
        net_info = self.pop[0].net_info
        self.pop[0].eval = log_data[3]
        self.pop[0].gene = np.array(log_data[5:]).reshape((net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()

    
    def num_epochs_scheduler(self, eval_num, max_eval, max_epochs, min_epochs=5):
        #intervals = np.arange(min_epochs, max_epochs + 1, 20)
        #num_epochs = min_epochs
        #for i in intervals:
        #    if i <= ((eval_num + 10)/max_eval)*max_epochs: num_epochs = i
        
        #return min(num_epochs, max_epochs)
        return min_epochs

    
    def get_fittest(self, individuals, mode='eval'):
        max_fitness = 0
        fittest = None
        for ind in individuals:
            if mode=='eval':
                if ind.eval - max_fitness >= self.epsilon:
                    max_fitness = ind.eval
                    fittest = ind
                elif abs(ind.eval - max_fitness) <= self.epsilon:
                    if fittest == None:
                        max_fitness = ind.eval
                        fittest = ind
                    elif ind.trainable_params < fittest.trainable_params:
                        max_fitness = ind.eval
                        fittest = ind
                
                if self.fittest == None:
                    self.fittest = fittest
                elif fittest != None:
                    if fittest.eval > self.fittest.eval:
                        self.fittest = fittest
            elif mode == 'novelty':
                if ind.novelty > max_fitness:
                    max_fitness = ind.novelty
                    fittest = ind
                if self.fittest == None:
                    self.fittest = fittest
                elif fittest.novelty > self.fittest.novelty:
                    self.fittest = fittest
                    if fittest not in self.search_archive: self.search_archive.append(fittest)
        
        return fittest

    
    def get_invalid_individuals(self):
        invalids = []
        for p in self.pop:
            if p.eval == 0: invalids.append(p)
        
        return invalids

    
    def novelty_survivor_selection(self, parents, children, tour_size):
        total_pool = parents + children
        next_gen = []

        while len(next_gen) < len(parents):
            tournament = np.random.choice(total_pool, tour_size, replace=False)
            fittest = self.get_fittest(tournament, mode='novelty')
            if fittest not in next_gen and fittest.novelty != 0:
                next_gen.append(fittest)

        for p in parents:
            if p in self.pop:
                self.pop.remove(p)

        for c in next_gen:
            self.pop.append(c)

        if self.fittest != None and self.fittest not in self.pop:
            self.pop.append(self.fittest)


    def survivor_selection(self, parents, children, tour_size):
        print('SURVIVOR SELECTION')
        current_epochs = children[0].epochs_trained
        to_retrain = []
        num_epochs_list = []
        
        for p in parents:
            if p.epochs_trained < current_epochs:
                to_retrain.append(p)
                num_epochs_list.append(current_epochs - p.epochs_trained)
        
        if len(to_retrain) > 0:
            self.retrain(to_retrain, num_epochs_list)

        #add invalid individuals to survivor selection process
        parents.extend(self.get_invalid_individuals())
        total_pool = parents + children
        next_gen = []

        while len(next_gen) < len(parents):
            tournament = np.random.choice(total_pool, tour_size, replace=False)
            fittest = self.get_fittest(tournament)
            if fittest != None:
                if fittest not in next_gen and fittest.eval != 0:
                    next_gen.append(fittest)

        for p in parents:
            if p in self.pop:
                self.pop.remove(p)

        for c in next_gen:
            self.pop.append(c)

        if self.fittest != None and self.fittest not in self.pop:
            self.pop.append(self.fittest)
    

    def tournament_selection(self, tour_pool, tour_size, num_tours, mode='eval'):
        print('PARENT SELECTION')
        selected = []
        while len(selected) < num_tours:
            tournament = np.random.choice(tour_pool, tour_size, replace=False)
            fittest = self.get_fittest(tournament, mode=mode)
            if fittest not in selected:
                if mode=='eval' and fittest.eval != 0:
                    selected.append(fittest)
                elif mode=='novelty' and fittest.novelty != 0:
                    selected.append(fittest)
        
        return selected

    # in case parents trained for fewer epochs than offspring, this function is called
    # it will look in the directory p_files for the pickled graph that was generated when initially training each individual
    def retrain(self, parent_pool, num_epochs_list):
        print('RETRAINING PARENTS')
        DAG_list = []
        model_names = []
        for p in parent_pool:
            assert(p.model_name)
            model_names.append(p.model_name)
            pickle_name = p.model_name.replace('.hdf5', '.gpickle')
            pickle_name = './p_files/' + pickle_name
            G = nx.read_gpickle(pickle_name)
            DAG_list.append(G)
        
        fp = self.eval_func(DAG_list, num_epochs_list, model_names, retrain = True)
        assert(len(parent_pool) == len(fp))
        
        for i in range(len(fp)):
            parent_pool[i].eval = fp[parent_pool[i].model_name][0]
            parent_pool[i].trainable_params = fp[parent_pool[i].model_name][1]
            parent_pool[i].epochs_trained = num_epochs_list[i]
    

    def get_stats(self, mode='eval'):
        evals = []
        for ind in self.pop:
            if mode == 'eval':
                if ind.eval is not None:
                    evals.append(ind.eval)
            elif mode == 'novelty':
                if ind.novelty is not None:
                    print('Individual: {}, Novelty: {}'.format(ind, ind.novelty))
                    evals.append(ind.novelty)
        
        return np.mean(evals), np.max(evals)
    

    def plot_evals(self, mean_evals, max_evals, mode='eval'):
        pickle.dump(mean_evals, open("mean_evals.p", "wb"))
        pickle.dump(max_evals, open("max_evals.p", "wb"))
        
        #mean_of_mean_evals = np.mean(mean_evals, axis=0)
        #std_of_mean_evals = np.std(mean_evals, axis=0)
        #mean_of_max_evals = np.mean(max_evals, axis=0)
        #std_of_max_evals = np.std(max_evals, axis=0)
        
        gens = []
        for i in range(len(mean_evals)):
            gens.append(i)
        
        plt.figure()
        plt.errorbar(x=gens, y=mean_evals)
        plt.errorbar(x=gens, y=max_evals)
        plt.title('Fitness vs Time')
        plt.xlabel('Generation')
        plt.ylabel('F1 Score')
        plt.legend(['Mean Fitness', 'Max Fitness'], loc='upper left')
        plt.savefig('cgpunet_drive_{}.png'.format(mode))
        plt.close()

    
    def evaluate_novelty(self, next_generation = None):
        k = int(math.sqrt(len(self.pop))) + 1
        DAGs = dict()
        
        for p in self.pop:
            DAGs[p] = cgp_2_dag(p.active_net_list())
        
        if next_generation is None:
            for individual in DAGs.keys():
                neighbours = get_neighbours(list(DAGs.values()), DAGs[individual], k)
                knn_val = 0
                for n in neighbours: knn_val += n[1]
                individual.novelty = 100 - 100*(knn_val/k)
        else:
            k_a = min(len(self.search_archive), k)
            DAGs_archive = []
            if k_a > 0:
                for a in self.search_archive:
                    DAGs_archive.append(cgp_2_dag(a.active_net_list()))
            for individual in next_generation:
                G = cgp_2_dag(individual.active_net_list())
                neighbours_current = get_neighbours(list(DAGs.values()), G, k)
                neighbours_archive = get_neighbours(DAGs_archive, G, k_a)

                knn_val = 0

                for n in neighbours_current: knn_val += n[1]
                for n in neighbours_archive: knn_val += n[1]

                individual.novelty = 100 - 100*(knn_val/(k + k_a))


    def check_memory(self, individual):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        arguments = (individual.active_net_list(), self.eval_func.batchsize, self.eval_func.input_shape, self.eval_func.target_shape, return_dict)
        p = multiprocessing.Process(target=get_model_memory_usage, args=arguments)
        p.start()
        p.join()

        mem = return_dict["memory"]
        
        return mem >= self.gpu_mem


    # Evolution CGP:
    #   At each iteration:
    #     - Generate lambda individuals in which at least one active node changes (i.e., forced mutation)
    #     - Mutate the best individual with neutral mutation (unchanging the active nodes)
    #         if the best individual is not updated.
    #TODO: change lam to pop_size
    def modified_evolution(self, mutation_rate=0.01, log_file='./log.txt', arch_file='./arch.txt', load_population='', init_gen=0, mode='eval'):
        
        def stopping_criteria(mode, mean_evals):
            if mode == 'eval':
                return self.num_gen < self.max_eval
            else:
                if self.num_gen < 15: 
                    return True
                else:
                    current_eval = mean_evals[len(mean_evals) - 1]
                    previous_eval = mean_evals[len(mean_evals) - 11]
                    return (current_eval - previous_eval)/10 > 1e-7
        
        with open('child.txt', 'w') as fw_c :
            writer_c = csv.writer(fw_c, lineterminator='\n')
            start_time = time.time()
            #eval_flag = np.empty(self.lam)
            num_tours = max(int(0.2*self.pop_size), 1)
            tour_size = min(5, len(self.pop))
            
            mean_evals = []
            max_evals = []

            if not load_population:
                #initialize and evaluate initial population
                print('GEN 0: INITIALIZING AND EVALUATING')
                print('Population: {}'.format(self.pop))
                for i in np.arange(0, len(self.pop), self.lam):
                    for j in range(i, min(i + self.lam, len(self.pop))):
                        active_num = self.pop[j].count_active_node()
                        pool_num = self.pop[j].check_pool()
                        while active_num < self.pop[j].net_info.min_active_num or pool_num > self.max_pool_num or self.check_memory(self.pop[j]):
                            self.pop[j].mutation(1.0)
                            active_num = self.pop[j].count_active_node()
                            pool_num= self.pop[j].check_pool()
                        self.pop[j].model_name = self.basename + '_' + str(self.num_gen) + '_' + str(j) + '.hdf5'
                
                if mode == 'eval':
                    self._evaluation(self.pop, [True]*len(self.pop), init_flag=True)
                else:
                    self.evaluate_novelty()
            else:
                self.load_population(load_population, init_gen)
            
            mean_fit, max_fit = self.get_stats(mode=mode)
            mean_evals.append(mean_fit)
            max_evals.append(max_fit)
            
            print('POPULATION INITIALIZED')
            print(self._log_data(net_info_type='active_only', start_time=start_time))

            while stopping_criteria(mode, mean_evals):
                self.pickle_population('./p_files_netlists', mode=mode)
                self.num_gen += 1
                print('GENERATION {}'.format(self.num_gen))
                parents = self.tournament_selection(self.pop, tour_size, num_tours, mode=mode)
                children = []
                eval_flag = np.empty(len(parents)*self.lam)

                # reproduction
                for i, p in enumerate(parents):
                    if p is None:
                        print('Nonetype individual - skipping {}'.format(p))
                        continue
                    for j in range(self.lam):
                        eval_flag[i*self.lam + j] = False
                        child = Individual(self.net_info, self.init)
                        child.copy(p)
                        active_num = child.count_active_node()
                        pool_num = child.check_pool()
                        # mutation (forced mutation)
                        while not eval_flag[i*self.lam + j] or active_num < child.net_info.min_active_num or pool_num > self.max_pool_num or self.check_memory(self.pop[j]):
                            child.copy(p)
                            eval_flag[i*self.lam + j] = child.mutation(mutation_rate)
                            active_num = child.count_active_node()
                            pool_num = child.check_pool()
                        
                        child.model_name = self.basename + '_' + str(self.num_gen) + '_' + str(i*self.lam+j) + '.hdf5'
                        children.append(child)
                    
                if mode == 'eval':
                    self._evaluation(children, eval_flag)
                    self.survivor_selection(parents, children, tour_size)
                else:
                    self.evaluate_novelty(next_generation=children)
                    self.novelty_survivor_selection(parents, children, tour_size)


                mean_fit, max_fit = self.get_stats(mode=mode)
                mean_evals.append(mean_fit)
                max_evals.append(max_fit)

                self.plot_evals(mean_evals, max_evals, mode)
                
                # save
                f = open('arch_child.txt', 'a')
                writer_f = csv.writer(f, lineterminator='\n')
                for c in range(1 + self.lam):
                    writer_c.writerow(self._log_data_children(net_info_type='full', start_time=start_time, pop=self.pop[c]))
                    writer_f.writerow(self._log_data_children(net_info_type='active_only', start_time=start_time, pop=self.pop[c]))
                f.close()

                # display and save log
                print(self._log_data(net_info_type='active_only', start_time=start_time))
                fw = open(log_file, 'a')
                writer = csv.writer(fw, lineterminator='\n')
                writer.writerow(self._log_data(net_info_type='full', start_time=start_time))
                fa = open('arch.txt', 'a')
                writer_a = csv.writer(fa, lineterminator='\n')
                writer_a.writerow(self._log_data(net_info_type='active_only', start_time=start_time))
                fw.close()
                fa.close()
            
            print('mean evals: {}'.format(mean_evals))
            print('max evals: {}'.format(max_evals))

        
