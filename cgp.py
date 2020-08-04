#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import time
import numpy as np
import math
import networkx as nx
from cgp_2_dag import *
import matplotlib.pyplot as plt
import pickle

# gene[f][c] f:function type, c:connection (nodeID)
class Individual(object):

    def __init__(self, net_info, init):
        self.net_info = net_info
        self.gene = np.zeros((self.net_info.node_num + self.net_info.out_num, self.net_info.max_in_num + 1)).astype(int)
        self.is_active = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.is_pool = np.empty(self.net_info.node_num + self.net_info.out_num).astype(bool)
        self.eval = None
        self.epochs_trained = 0
        self.gen_num = 0
        self.model_name = ''
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
        is_pool = True
        pool_num = 0
        for n in range(self.net_info.node_num + self.net_info.out_num):
            if self.is_active[n]:
                if self.gene[n][0] > 19:
                    is_pool = False
                    pool_num += 1
        return is_pool, pool_num

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
    def __init__(self, net_info, eval_func, max_eval, pop_size=100, lam=4, imgSize=256, init=False):
        self.lam = lam
        self.net_info = net_info
        self.pop_size = pop_size
        self.pop = [Individual(self.net_info, init) for _ in range(1 + self.pop_size)]
        self.eval_func = eval_func
        self.num_gen = 0
        self.num_eval = 0
        self.max_pool_num = int(math.log2(imgSize) - 2)
        self.max_eval = max_eval
        self.init = init
        self.fittest = None
    

    def pickle_population(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        p_name = os.path.join(save_dir, 'population_' + str(self.num_gen) + '.p')

        try:
            pickle.dump(self.pop, open(p_name, "wb"))
        except:
            pass


    def _evaluation(self, pop, eval_flag):
        # create network list
        DAG_list = []
        #net_lists = []
        active_index = np.where(eval_flag)[0]
        for i in active_index:
            G = cgp_2_dag(pop[i].active_net_list())
            DAG_list.append(G)

        num_epochs = self.num_epochs_scheduler(self.num_eval, self.max_eval, self.eval_func.epoch_num)
        num_epochs_list = [num_epochs]*len(DAG_list)
        
        # evaluation
        fp, model_names = self.eval_func(DAG_list, num_epochs_list, self.num_gen)
        for i, j in enumerate(active_index):
            pop[j].gen_num = self.num_gen
            pop[j].eval = fp[i]
            pop[j].epochs_trained = num_epochs
            pop[j].model_name = model_names[i]
        evaluations = np.zeros(len(pop))
        for i in range(len(pop)):
            evaluations[i] = pop[i].eval

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
        intervals = np.arange(min_epochs, max_epochs + 1, 20)
        num_epochs = min_epochs
        for i in intervals:
            if i <= ((eval_num + 10)/max_eval)*max_epochs: num_epochs = i
        
        return min(num_epochs, max_epochs)

    
    def get_fittest(self, individuals):
        max_fitness = 0
        fittest = None
        for ind in individuals:
            assert(ind.eval != None)
            if ind.eval > max_fitness:
                max_fitness = ind.eval
                fittest = ind
                if self.fittest == None:
                    self.fittest = fittest
                elif fittest.eval > self.fittest.eval:
                    self.fittest = fittest
        
        return fittest
    

    def survivor_selection(self, parents, children, tour_size):
        current_epochs = children[0].epochs_trained
        to_retrain = []
        num_epochs_list = []
        
        for p in parents:
            if p.epochs_trained < current_epochs:
                to_retrain.append(p)
                num_epochs_list.append(current_epochs - p.epochs_trained)
        
        if len(to_retrain) > 0:
            self.retrain(to_retrain, num_epochs_list)

        total_pool = parents + children
        next_gen = []

        while len(next_gen) < len(parents):
            tournament = np.random.choice(total_pool, tour_size, replace=False)
            fittest = self.get_fittest(tournament)
            if fittest not in next_gen:
                next_gen.append(fittest)

        for p in parents:
            if p in self.pop:
                self.pop.remove(p)

        for c in next_gen:
            self.pop.append(c)

        if self.fittest != None and self.fittest not in self.pop:
            self.pop.append(self.fittest)
    

    def tournament_selection(self, tour_pool, tour_size, num_tours):
        selected = []
        while len(selected) < num_tours:
            tournament = np.random.choice(tour_pool, tour_size, replace=False)
            fittest = self.get_fittest(tournament)
            if fittest not in selected:
                selected.append(fittest)
        
        return selected

    # in case parents trained for fewer epochs than offspring, this function is called
    # it will look in the directory p_files for the pickled graph that was generated when initially training each individual
    def retrain(self, parent_pool, num_epochs_list):
        DAG_list = []
        model_names = []
        for p in parent_pool:
            assert(p.model_name)
            model_names.append(p.model_name)
            pickle_name = p.model_name.replace('.hdf5', '.gpickle')
            pickle_name = './p_files/' + pickle_name
            G = nx.read_gpickle(pickle_name)
            DAG_list.append(G)
        
        fp, _ = self.eval_func(DAG_list, num_epochs_list, model_names, retrain = True)
        assert(len(parent_pool) == len(fp))
        
        for i in range(len(fp)):
            parent_pool[i].eval = fp[i]
            parent_pool[i].epochs_trained = num_epochs_list[i]
    

    def get_fitness_stats(self):
        evals = []
        for ind in self.pop:
            evals.append(ind.eval)
        
        return np.mean(evals), np.max(evals)

    # Evolution CGP:
    #   At each iteration:
    #     - Generate lambda individuals in which at least one active node changes (i.e., forced mutation)
    #     - Mutate the best individual with neutral mutation (unchanging the active nodes)
    #         if the best individual is not updated.
    #TODO: change lam to pop_size
    def modified_evolution(self, mutation_rate=0.01, log_file='./log.txt', arch_file='./arch.txt'):
        with open('child.txt', 'w') as fw_c :
            writer_c = csv.writer(fw_c, lineterminator='\n')
            start_time = time.time()
            eval_flag = np.empty(self.lam)
            active_num = []
            pool_num = []
            num_tours = int(0.2*self.pop_size)
            tour_size = 5
            
            mean_evals = []
            max_evals = []

            #initialize and evaluate initial population
            print('GEN 0: INITIALIZING AND EVALUATING')
            for i in np.arange(0, len(self.pop), self.lam):
                for j in range(i, i + self.lam):
                    active_num[j] = self.pop[j].count_active_node()
                    _, pool_num = self.pop[i].check_pool()
                    while active_num[j] < self.pop[j].net_info.min_active_num or pool_num > self.max_pool_num:
                        self.pop[j].mutation(1.0)
                        active_num[j] = self.pop[j].count_active_node()
                        _, pool_num= self.pop[j].check_pool() 
                self._evaluation([self.pop[i:j]], np.full((self.lam,), True))
            
            mean_fit, max_fit = self.get_fitness_stats()
            mean_evals.append(mean_fit)
            max_evals.append(max_fit)
            
            print('POPULATION INITIALIZED')
            print(self._log_data(net_info_type='active_only', start_time=start_time))

            while self.num_gen < self.max_eval:
                self.num_gen += 1
                print('GENERATION {}'.format(self.num_gen))
                parents = self.tournament_selection(self.pop, tour_size, num_tours)
                children = []

                # reproduction
                for p in parents:
                    p_children = []
                    for i in range(self.lam):
                        eval_flag[i] = False
                        child = Individual(self.net_info, self.init)
                        child.copy(p)
                        active_num = child.count_active_node()
                        _, pool_num = child.check_pool()
                        # mutation (forced mutation)
                        while not eval_flag[i] or active_num < child.net_info.min_active_num or pool_num > self.max_pool_num:
                            child.copy(p)
                            eval_flag[i] = child.mutation(mutation_rate)
                            active_num = child.count_active_node()
                            _, pool_num = child.check_pool()
                        
                        p_children.append(child)
                    
                    self._evaluation(p_children, eval_flag)
                    children.extend(p_children)

                self.survivor_selection(parents, children, tour_size)

                mean_fit, max_fit = self.get_fitness_stats()
                mean_evals.append(mean_fit)
                max_evals.append(max_fit)
                
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

        
        pickle.dump(mean_evals, open("mean_evals.p", "wb"))
        pickle.dump(max_evals, open("max_evals.p", "wb"))
        
        mean_of_mean_evals = np.mean(mean_evals, axis=0)
        std_of_mean_evals = np.std(mean_evals, axis=0)

        mean_of_max_evals = np.mean(max_evals, axis=0)
        std_of_max_evals = np.std(max_evals, axis=0)
        
        gens = []
        for i in range(len(mean_evals)):
            gens.append(i)
        
        plt.figure()
        plt.errorbar(x=gens, y=mean_of_mean_evals, yerr=std_of_mean_evals)
        plt.errorbar(x=gens, y=mean_of_max_evals, yerr=std_of_max_evals)
        plt.title('Fitness vs Time')
        plt.xlabel('Generation')
        plt.ylabel('F1 Score')
        plt.legend(['Mean Fitness', 'Max Fitness'], loc='upper left')
        plt.savefig('cgpunet_drive_fitness.png')