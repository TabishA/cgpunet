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
import random
from knn import calculate_distance, check_local_neighbor, dag_2_vec
from ged import get_distances_simgnn
from tensorflow.compat.v1.keras import backend as K
import multiprocessing
from simgnn.src.parser import parameter_parser


def get_approximate_model_memory(net_list, batch_size, input_shape, return_dict):
    G = cgp_2_dag(net_list)
    vecs = dag_2_vec(G, input_size=(input_shape[0], input_shape[1]))

    memsum = 0
    params = 0

    for v in vecs:
        if v[0] == 0:
            memsum += v[1] * v[2]
        else:
            memsum += v[0] * v[1] * v[2]

    for i in range(1, len(vecs)):
        v = vecs[i]
        v_prev = vecs[i - 1]
        if v_prev[0] == 0:
            params += v[2] * v_prev[2]
        else:
            params += v_prev[0] * v_prev[1] * v_prev[2] * (v[1] * v[2])

    total = batch_size * memsum + params

    return_dict["memory"] = np.round(total / (1024 ** 3), 3)


# gene[f][c] f:function type, c:connection (nodeID)
class Individual(object):
    #Ind 2_json function here returning the graph edge list and its labels
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
        self.model_mem = 0
        self.labels = {'input': '0', 'ConvBlock': '1', 'ResBlock': '2', 'DeconvBlock': '3', 'Concat': '4', 'Sum': '5', \
                  'Avg_Pool': '6', 'Max_Pool': '7'}
        if init:
            print('init with specific architectures')
            self.init_gene_with_conv()  # In the case of starting only convolution
        else:
            self.init_gene()  # generate initial individual randomly

    def init_gene_with_conv(self):
        # initial architecture
        arch = ['S_ConvBlock_64_3']

        input_layer_num = int(self.net_info.input_num / self.net_info.rows) + 1
        output_layer_num = int(self.net_info.out_num / self.net_info.rows) + 1
        layer_ids = [((self.net_info.cols - 1 - input_layer_num - output_layer_num) + i) // (len(arch)) for i in
                     range(len(arch))]
        prev_id = 0  # i.e. input layer
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
                # print('max: {}, min: {}, gene: {}'.format(max_connect_id, min_connect_id, self.gene[n][2]))

        self.check_active()

    def __check_course_to_out(self, n):
        if not self.is_active[n]:
            self.is_active[n] = True
            t = self.gene[n][0]
            in_num = self.net_info.func_in_num[t]

            for i in range(in_num):
                if self.gene[n][i + 1] >= self.net_info.input_num:
                    self.__check_course_to_out(self.gene[n][i + 1] - self.net_info.input_num)

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
        neutral_check = False

        for n in range(self.net_info.node_num + self.net_info.out_num):
            t = self.gene[n][0]
            # mutation for type gene
            type_num = self.net_info.func_type_num if n < self.net_info.node_num else self.net_info.out_type_num
            if np.random.rand() < mutation_rate and type_num > 1:
                self.gene[n][0] = self.__mutate(self.gene[n][0], 0, type_num)
                neutral_check = True
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
                    self.gene[n][i + 1] = self.__mutate(self.gene[n][i + 1], min_connect_id, max_connect_id)
                    neutral_check = True
                    if self.is_active[n] and i < in_num:
                        active_check = True

        self.check_active()
        print("Active Node Mutation is {}".format(active_check))
        print("Neutral Mutation is {}".format(neutral_check))
        return active_check, neutral_check

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
                    self.gene[n][i + 1] = self.__mutate(self.gene[n][i + 1], min_connect_id, max_connect_id)

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
        self.model_mem = source.model_mem

    def active_net_list(self):
        net_list = [["input", 0, 0]]
        active_cnt = np.arange(self.net_info.input_num + self.net_info.node_num + self.net_info.out_num)
        active_cnt[self.net_info.input_num:] = np.cumsum(self.is_active)

        for n, is_a in enumerate(self.is_active):
            if is_a:
                t = self.gene[n][0]
                if n < self.net_info.node_num:  # intermediate node
                    type_str = self.net_info.func_type[t]
                else:  # output node
                    type_str = self.net_info.out_type[t]

                connections = [active_cnt[self.gene[n][i + 1]] for i in range(self.net_info.max_in_num)]
                net_list.append([type_str] + connections)

        return net_list


    def ind_2_dict(self):
        ind_edgelist = []
        ind_labels = []

        netlist = self.active_net_list()
        G = cgp_2_dag(netlist)

        if nx.isolates(G):
            G.remove_nodes_from(nx.isolates(G))

        nodelist = list(G.nodes)

        for node in nodelist:
            elems = node.split('_')
            fn = G.nodes[node]['function']
            id = int(G.nodes[node]['id'])
            if fn == 'input':
                ind_labels.append(self.labels[fn])
                continue
            elif fn in ['Sum', 'Concat']:
                n1 = int(elems[1])
                n2 = int(elems[2])
                edge1 = [n1, id]
                edge2 = [n2, id]
                ind_edgelist.append(edge1)
                ind_edgelist.append(edge2)
                ind_labels.append(self.labels[fn])
            else:
                n1 = int(elems[len(elems) - 2])
                ind_edgelist.append([n1, id])
                ind_labels.append(self.labels[fn])

        return {'graph': ind_edgelist, 'labels': ind_labels, 'modelname': self.model_name}


# CGP with (1 + \lambda)-ES
class CGP(object):
    def __init__(self, net_info, eval_func, max_eval, pop_size=100, lam=4, gpu_mem=16, imgSize=256, init=False,
                 basename='cgpunet_drive'):
        self.lam = lam
        # GPU memory in GB
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
        self.novel = None
        self.basename = basename
        self.search_archive = []
        self.epsilon = 0.05
        self.args = parameter_parser()
        self.simgnn_labels = pickle.load(open(self.args.node_labels_path, 'rb'))


    def pickle_population(self, save_dir, mode='eval'):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        p_name = os.path.join(save_dir, 'population_' + mode + '_' + str(self.num_gen) + '.p')

        try:
            pickle.dump(self.pop, open(p_name, "wb"))
        except:
            pass

    def pickle_state_annealing(self, save_dir, candidate_number, current_temp, max_temp, final_temp, alpha, cooling,
                               consider_neutral, log_dir, mutation_rate, eval_num):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        p_name = os.path.join(save_dir, 'annealing_' + str(current_temp) + '.p')

        try:
            pickle.dump([self.pop, candidate_number, current_temp, max_temp, final_temp, alpha, cooling,
                         consider_neutral, log_dir, mutation_rate, eval_num], open(p_name, "wb"))
        except:
            pass

    def pickle_state_shc(self, save_dir, candidate_number, current_iter, maxIter, consider_neutral, log_dir,
                         mutation_rate, eval_num, ):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        p_name = os.path.join(save_dir, 'shc_' + str(candidate_number) + '_' + str(current_iter) + '.p')

        try:
            pickle.dump([self.pop, candidate_number, current_iter, maxIter,
                         consider_neutral, log_dir, mutation_rate, eval_num], open(p_name, "wb"))
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
        num_epochs_list = [num_epochs] * len(DAG_list)

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
            if self.fittest != None:
                if pop[i].eval > self.fittest.eval:
                    self.fittest = pop[i]

        if not init_flag:
            self.num_eval += len(DAG_list)
        return evaluations

    def _log_data(self, net_info_type='active_only', start_time=0):
        log_list = [self.num_gen, self.num_eval, time.time() - start_time, self.pop[0].eval,
                    self.pop[0].count_active_node()]
        if net_info_type == 'active_only':
            log_list.append(self.pop[0].active_net_list())
        elif net_info_type == 'full':
            log_list += self.pop[0].gene.flatten().tolist()
        else:
            pass
        return log_list

    def _log_state(self, log_dir, eval_num, index, current_iter, neighborFitness=[0.0] * 8, timeElapsed=[0.0] * 8,
                   probability=[0.0] * 8):
        dir_path = os.path.join(os.getcwd(), log_dir)
        if not os.path.isdir(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as e:
                print(e)
                return

        fa = open(os.path.join(dir_path, 'active_netlist.txt'), 'a')
        writera = csv.writer(fa, lineterminator='\n')
        ff = open(os.path.join(dir_path, 'full_netlist.txt'), 'a')
        writerf = csv.writer(ff, lineterminator='\n')
        fspecs = open(os.path.join(dir_path, 'run_specifications.txt'), 'a')
        writerspecs = csv.writer(fspecs, lineterminator='\n')
        ffit = open(os.path.join(dir_path, 'fitness_trend.txt'), 'a')
        writerfit = csv.writer(ffit, lineterminator='\n')

        candidate_fitness = []
        candidate_memory = []
        for i in range(self.lam):
            candidate_fitness.append(self.pop[index + i].eval)
            candidate_memory.append(self.pop[index + i].model_mem)
            netlist = [i + index, current_iter, self.pop[i].active_net_list()]
            writera.writerow(netlist)
            full_netlist = [i + index, current_iter, self.pop[i].gene.flatten().tolist()]
            writerf.writerow(full_netlist)

        specs = [current_iter] + candidate_fitness
        writerfit.writerow(specs)
        for x in neighborFitness:
            specs.append(x)
        for x in candidate_memory:
            specs.append(x)
        for x in probability:
            specs.append(x)
        for x in timeElapsed:
            specs.append(x)
        specs.append(eval_num)
        writerspecs.writerow(specs)

        fa.close()
        ff.close()
        fspecs.close()
        ffit.close()

        return specs[0:5]

    def _log_data_children(self, net_info_type='active_only', start_time=0.0, pop=None):
        log_list = [self.num_gen, self.num_eval, time.time() - start_time, pop.eval, pop.count_active_node()]
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
        self.pop[0].gene = np.array(log_data[5:]).reshape(
            (net_info.node_num + net_info.out_num, net_info.max_in_num + 1))
        self.pop[0].check_active()

    def num_epochs_scheduler(self, eval_num, max_eval, max_epochs, min_epochs=5):
        # intervals = np.arange(min_epochs, max_epochs + 1, 20)
        # num_epochs = min_epochs
        # for i in intervals:
        #    if i <= ((eval_num + 10)/max_eval)*max_epochs: num_epochs = i

        # return min(num_epochs, max_epochs)
        return min_epochs

    def get_fittest(self, individuals, mode='eval'):
        max_fitness = 0
        max_novelty = 0
        fittest = None
        novel = None
        for ind in individuals:
            if mode == 'eval':
                if ind.eval - max_fitness >= self.epsilon:
                    max_fitness = ind.eval
                    fittest = ind
                elif abs(ind.eval - max_fitness) <= self.epsilon:
                    if fittest == None:
                        max_fitness = ind.eval
                        fittest = ind
                    elif ind.trainable_params < fittest.trainable_params or ind.novelty > fittest.novelty:
                        max_fitness = ind.eval
                        fittest = ind

                if self.fittest == None:
                    self.fittest = fittest
                elif fittest != None:
                    if fittest.eval > self.fittest.eval:
                        self.fittest = fittest

                if ind.novelty > max_novelty:
                    max_novelty = ind.novelty
                    novel = ind
                if self.novel == None:
                    self.novel = novel
                elif novel.novelty > self.novel.novelty:
                    self.novel = novel
                    if novel not in self.search_archive: self.search_archive.append(novel)

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

        # add invalid individuals to survivor selection process
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
                if mode == 'eval' and fittest.eval != 0:
                    selected.append(fittest)
                elif mode == 'novelty' and fittest.novelty != 0:
                    selected.append(fittest)

        return selected

    # in case parents trained for fewer epochs than offspring, this function is called
    # it will look in the directory p_files for the pickled graph that was generated when initially training each individual
    def retrain(self, parent_pool, num_epochs_list):
        print('RETRAINING PARENTS')
        DAG_list = []
        model_names = []
        for p in parent_pool:
            assert (p.model_name)
            model_names.append(p.model_name)
            pickle_name = p.model_name.replace('.hdf5', '.gpickle')
            pickle_name = './p_files/' + pickle_name
            G = nx.read_gpickle(pickle_name)
            DAG_list.append(G)

        fp = self.eval_func(DAG_list, num_epochs_list, model_names, retrain=True)
        assert (len(parent_pool) == len(fp))

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

        # mean_of_mean_evals = np.mean(mean_evals, axis=0)
        # std_of_mean_evals = np.std(mean_evals, axis=0)
        # mean_of_max_evals = np.mean(max_evals, axis=0)
        # std_of_max_evals = np.std(max_evals, axis=0)

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
        k = int(math.sqrt(len(self.search_archive) + len(self.pop)))
        if k % 2 == 0:
            k += 1
        
        if next_generation is None:
            next_generation = self.pop
            
        distances_dict = dict()

        for j, individual in enumerate(next_generation):
            all_DAGs = self.pop.copy()
                
            if individual.model_name in distances_dict.keys():
                for ind in self.pop:
                    if ind.model_name in distances_dict[individual.model_name].keys():
                        #print('removing individual from comparison list')
                        all_DAGs.remove(ind)
                
            #distances = get_distances(all_DAGs + self.search_archive, individual)
            pop_dict = list()
            individual_dict = individual.ind_2_dict() #G1

            for ind in (all_DAGs + self.search_archive):
                pop_dict.append(ind.ind_2_dict())

            pairs_list = []

            for graph2 in pop_dict:
                data = dict()
                data['graph_1'] = individual_dict['graph']
                data['labels_1'] = individual_dict['labels']
                data['modelname1'] = individual_dict['modelname']
                data['graph_2'] = graph2['graph']
                data['labels_2'] = graph2['labels']
                data['modelname2'] = graph2['modelname']
                pairs_list.append(data)

            # distances = get_distances_simgnn(simgnn_model_path=self.args.load_path, global_labels=self.simgnn_labels, data=pairs_list)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()

            arguments = (self.args.load_path, self.simgnn_labels, pairs_list, return_dict)
            p = multiprocessing.Process(target=get_distances_simgnn, args=arguments)
            p.start()
            p.join()

            distances = return_dict["distances"]

            for d in distances:
                entry = {individual.model_name: distances[d]}
                if d in distances_dict.keys():
                    distances_dict[d].update(entry)
                else:
                    distances_dict[d] = entry

            if individual.model_name in distances_dict.keys():
                distances_dict[individual.model_name].update(distances)
                #print('updating entry for current individual: {}'.format(distances_dict[individual.model_name]))
            else:
                distances_dict[individual.model_name] = distances
                #print('new entry for current individual: {}'.format(distances))
            
            #print('length of distances_dict.keys() = {}'.format(len(distances_dict.keys())))
            #print(distances_dict)

        for individual in next_generation:
            if not individual.model_name in list(distances_dict.keys()): continue
            distances = distances_dict[individual.model_name]
            distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
            knn_val = 0

            for i, d in enumerate(distances):
                if i == 0: continue

                knn_val += distances[d]
                if i == k + 1: break
                
            individual.novelty = knn_val/k
            print('k = {}'.format(i-1))
            print('novelty = {}'.format(individual.novelty))
        
        pickle.dump(self.search_archive, open('search_archive.p', 'wb'))

    def check_memory(self, individual):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        arguments = (
        individual.active_net_list(), self.eval_func.batchsize, self.eval_func.input_shape, return_dict)
        p = multiprocessing.Process(target=get_approximate_model_memory, args=arguments)
        p.start()
        p.join()

        mem = return_dict["memory"]
        individual.model_mem = mem
        print('Model memory exceeds gpu memory : {}'.format(mem >= self.gpu_mem))
        return mem >= self.gpu_mem

    # Evolution CGP:
    #   At each iteration:
    #     - Generate lambda individuals in which at least one active node changes (i.e., forced mutation)
    #     - Mutate the best individual with neutral mutation (unchanging the active nodes)
    #         if the best individual is not updated.
    def modified_evolution(self, mutation_rate=0.01, log_file='./log.txt', arch_file='./arch.txt', load_population='',
                           init_gen=0, mode='eval'):

        def stopping_criteria(mode, mean_evals):
            if mode == 'eval':
                return self.num_gen < self.max_eval
            else:
                if self.num_gen < 15:
                    return True
                else:
                    current_eval = mean_evals[len(mean_evals) - 1]
                    previous_eval = mean_evals[len(mean_evals) - 11]
                    return (current_eval - previous_eval) / 10 > 1e-7

        with open('child.txt', 'w') as fw_c:
            writer_c = csv.writer(fw_c, lineterminator='\n')
            start_time = time.time()
            # eval_flag = np.empty(self.lam)
            num_tours = max(int(0.2 * self.pop_size), 1)
            tour_size = min(5, len(self.pop))

            mean_evals = []
            max_evals = []

            if not load_population:
                # initialize and evaluate initial population
                print('GEN 0: INITIALIZING AND EVALUATING')
                print('Population: {}'.format(self.pop))
                for i in np.arange(0, len(self.pop), self.lam):
                    for j in range(i, min(i + self.lam, len(self.pop))):
                        active_num = self.pop[j].count_active_node()
                        pool_num = self.pop[j].check_pool()
                        while active_num < self.pop[
                            j].net_info.min_active_num or pool_num > self.max_pool_num or self.check_memory(
                                self.pop[j]):
                            self.pop[j].mutation(1.0)
                            active_num = self.pop[j].count_active_node()
                            pool_num = self.pop[j].check_pool()
                        self.pop[j].model_name = self.basename + '_' + str(self.num_gen) + '_' + str(j) + '.hdf5'

                if mode == 'eval':
                    self._evaluation(self.pop, [True] * len(self.pop), init_flag=True)
                    self.evaluate_novelty()
                else:
                    self.evaluate_novelty()
            else:
                self.load_population(load_population, init_gen)
                self.search_archive = pickle.load(open('search_archive.p'), 'rb')
                mean_evals = pickle.load(open('./mean_evals.p'), 'rb')
                max_evals = pickle.load(open('./max_evals.p'), 'rb')

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
                eval_flag = np.empty(len(parents) * self.lam)

                # reproduction
                for i, p in enumerate(parents):
                    if p is None:
                        print('Nonetype individual - skipping {}'.format(p))
                        continue
                    for j in range(self.lam):
                        eval_flag[i * self.lam + j] = False
                        child = Individual(self.net_info, self.init)
                        child.copy(p)
                        active_num = child.count_active_node()
                        pool_num = child.check_pool()
                        # mutation (forced mutation)
                        while not eval_flag[
                            i * self.lam + j] or active_num < child.net_info.min_active_num or pool_num > self.max_pool_num or self.check_memory(
                                self.pop[j]):
                            child.copy(p)
                            eval_flag[i * self.lam + j], _ = child.mutation(mutation_rate)
                            active_num = child.count_active_node()
                            pool_num = child.check_pool()

                        child.model_name = self.basename + '_' + str(self.num_gen) + '_' + str(
                            i * self.lam + j) + '.hdf5'
                        children.append(child)

                if mode == 'eval':
                    self._evaluation(children, eval_flag)
                    self.evaluate_novelty(next_generation=children)
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
                    writer_c.writerow(
                        self._log_data_children(net_info_type='full', start_time=start_time, pop=self.pop[c]))
                    writer_f.writerow(
                        self._log_data_children(net_info_type='active_only', start_time=start_time, pop=self.pop[c]))
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

    def get_local_neighbor(self, ind, candidate_number, j, mut_rate, DAGs, consider_neutral=False):
        eval_flag = False
        mutated = False
        neighbor_not_found = True
        attempts = 0

        while neighbor_not_found:
            if attempts > 200:
                print("Mutant could not be generate by active or neutral mutations")
                ind.copy(self.pop[candidate_number + j])
                eval_flag = False
                break;

            else:
                print("Attempt {}".format(attempts[j]))
                ind.copy(self.pop[candidate_number + j])
                eval_flag, mutated = ind.mutation(mut_rate)
                active_num = ind.count_active_node()
                pool_num = ind.check_pool()

                if consider_neutral or attempts > 150:
                    neighbor_not_found = not mutated or active_num < ind.net_info.min_active_num or \
                                         pool_num > self.max_pool_num or self.check_memory(ind)
                else:
                    neighbor_not_found = not eval_flag or active_num < ind.net_info.min_active_num or \
                                         pool_num > self.max_pool_num or self.check_memory(ind)

                # If the generated neighbor is valid, then check if it is a local neighbor
                if not neighbor_not_found:
                    print("Checking Neighborhood")
                    DAG_ind = dict()
                    DAG_ind[ind] = cgp_2_dag(ind.active_net_list())
                    if not check_local_neighbor(list(DAGs.values()), DAGs[self.pop[candidate_number + j]],
                                                DAG_ind[ind]):
                        print("Mutant is not a local neighbor")
                        neighbor_not_found = True
            attempts += 1

        if mutated and not eval_flag:
            ind.eval = self.pop[candidate_number + j].eval
            ind.epochs_trained = self.pop[candidate_number + j].epochs_trained
            ind.trainable_params = self.pop[candidate_number + j].trainable_params

        return ind, eval_flag, attempts

    def simulated_annealing(self, candidate_index=0, mutation_rate=0.01, start_temp=100, final_temp=0.5,
                            alpha=0.5, log_dir='SA', load=False, load_file='', timestamp='',
                            consider_neutral=False, cooling='linear'):

        log_dir = log_dir + str(mutation_rate) + '_' + timestamp
        current_temp = start_temp
        eval_num = 0

        if load:
            loaded_state = pickle.load(open(load_file, "rb"))
            self.pop = loaded_state[0]
            self.pop_size = len(self.pop)
            candidate_index = loaded_state[1]
            start_temp = loaded_state[3]
            final_temp = loaded_state[4]
            alpha = loaded_state[5]
            cooling = loaded_state[6]
            if cooling == 'linear':
                current_temp = loaded_state[2] - alpha
            elif cooling == 'exp':
                current_temp = loaded_state[2] * math.pow(alpha, 1)
            consider_neutral = loaded_state[7]
            log_dir = loaded_state[8]
            mutation_rate = loaded_state[9]
            eval_num = loaded_state[10]

        else:
            # Create directories
            if not os.path.isdir(os.path.join(os.getcwd(), log_dir)):
                try:
                    os.makedirs(os.path.join(os.getcwd(), log_dir))
                except OSError as e:
                    print(e)
                    return
            if not os.path.isdir(os.path.join(os.getcwd(), log_dir + "/models")):
                try:
                    os.makedirs(os.path.join(os.getcwd(), log_dir + "/models"))
                except OSError as e:
                    print(e)
                    return

        print("log Dir {}".format(log_dir))
        for candidate_number in range(candidate_index, len(self.pop), self.lam):
            while current_temp > final_temp:
                print("Current Temp is {}".format(current_temp))
                neighbors = []
                process_time = []
                attempts = [0, 0, 0, 0]
                eval_flag = np.empty(self.lam, dtype=bool)  # To keep track active mutation
                DAGs = dict()
                for p in self.pop:
                    print("Fitness {}".format(p.eval))
                    DAGs[p] = cgp_2_dag(p.active_net_list())

                for j in range(min(self.lam, len(self.pop) - candidate_number)):
                    print("Currently Processing candidate number {}".format(candidate_number + j))
                    start_time = time.time()
                    mutant, eval_flag[j], attempts[j] = self.get_local_neighbor(Individual(self.net_info, self.init),
                                                                                candidate_number, j, mutation_rate,
                                                                                DAGs, consider_neutral)
                    mutant.model_name = log_dir + "/models/" + self.basename + '_' + str(candidate_number + j) + '_' + \
                                        str(current_temp) + '.hdf5'
                    neighbors.append(mutant)
                    process_time.append(time.time() - start_time)

                # Evaluate the generated mutants
                print("Evaluating the neighbors")
                eval_num += sum(eval_flag)
                neighbor_fitness = self._evaluation(neighbors, eval_flag)
                print("Summary of Neighbors generated")
                print("Fitness : {}".format(neighbor_fitness))
                print("Model Memory : {}".format([neighbors[x].model_mem for x in range(len(neighbors))]))
                print("Attempts to generate mutant : {}".format(attempts))

                # Check if neighbor is best so far
                prob = [1.0, 1.0, 1.0, 1.0]
                for index in range(self.lam):
                    if neighbor_fitness[index] != 0:

                        cost_diff = neighbor_fitness[index] - self.pop[candidate_number + index].eval

                        # if the new solution is better, accept it
                        if cost_diff > 0:
                            self.pop[candidate_number + index] = neighbors[index]
                        # if the new solution is not better, accept it with a probability of e^(-cost/temp)
                        else:
                            r = random.uniform(0, 1)
                            ac = math.exp(cost_diff / (current_temp / start_temp))
                            prob[index] = ac
                            print('Cost difference : {}'.format(cost_diff))
                            print("{} < {}".format(r, ac))
                            if r < ac:
                                self.pop[candidate_number + index] = neighbors[index]
                    else:
                        eval_num -= 1

                # display and save log
                print(self._log_state(log_dir, eval_num, index=candidate_number, current_iter=current_temp,
                                      neighborFitness=neighbor_fitness, timeElapsed=process_time, probability=prob))

                # save the current population
                self.pickle_state_annealing(log_dir + '/sa_files_netlists', candidate_number, current_temp, start_temp,
                                            final_temp, alpha, cooling, consider_neutral, log_dir, mutation_rate,
                                            eval_num)

                # decrement the temperature
                if cooling == 'linear':
                    current_temp -= alpha
                elif cooling == 'exp':
                    current_temp = current_temp * math.pow(alpha, 1)
            current_temp = start_temp

    def stochastic_hill_climbing(self, candidate_index=0, mutation_rate=0.01, maxIter=200, log_dir='SHC', load=False,
                                 load_file='', timestamp='', consider_neutral=False):

        log_dir = log_dir + str(mutation_rate) + '_' + timestamp
        current_iter = 0
        eval_num = 0

        if load:
            loaded_state = pickle.load(open(load_file, "rb"))
            self.pop = loaded_state[0]
            self.pop_size = len(self.pop)
            candidate_index = loaded_state[1]
            current_iter = loaded_state[2]
            maxIter = loaded_state[3]
            consider_neutral = loaded_state[4]
            log_dir = loaded_state[5]
            mutation_rate = loaded_state[6]
            eval_num = loaded_state[7]

        else:
            # Create directories
            if not os.path.isdir(os.path.join(os.getcwd(), log_dir)):
                try:
                    os.makedirs(os.path.join(os.getcwd(), log_dir))
                    os.makedirs(os.path.join(os.getcwd(), log_dir + "/models"))
                except OSError as e:
                    print(e)
                    return

        print("log Dir {}".format(log_dir))

        for candidate_number in range(candidate_index, len(self.pop), self.lam):
            while current_iter < maxIter:
                print("Current Iteration is {}".format(current_iter))
                neighbors = []
                process_time = []
                attempts = [0, 0, 0, 0]
                eval_flag = np.empty(self.lam, dtype=bool)  # To keep track active mutation
                DAGs = dict()

                # Get the dag list of the current population
                for p in self.pop:
                    print("Fitness {}".format(p.eval))
                    DAGs[p] = cgp_2_dag(p.active_net_list())

                # Generate local neighbors for the candidate solution
                for j in range(min(self.lam, len(self.pop) - candidate_number)):
                    print("Currently Processing candidate number {}".format(candidate_number + j))
                    start_time = time.time()
                    mutant, eval_flag[j], attempts[j] = self.get_local_neighbor(Individual(self.net_info, self.init),
                                                                                candidate_number, j, mutation_rate,
                                                                                DAGs, consider_neutral)
                    mutant.model_name = log_dir + "/models/" + self.basename + '_' + str(candidate_number + j) + '_' + \
                                        str(current_iter) + '.hdf5'
                    neighbors.append(mutant)
                    process_time.append(time.time() - start_time)

                # Evaluate the generated mutants
                print("Evaluating the neighbors")
                eval_num += sum(eval_flag)
                neighbor_fitness = self._evaluation(neighbors, eval_flag)
                print("Summary of Neighbors generated")
                print("Fitness : {}".format(neighbor_fitness))
                print("Model Memory : {}".format([neighbors[x].model_mem for x in range(len(neighbors))]))
                print("Attempts to generate mutant : {}".format(attempts))

                # Check if neighbor is best so far
                prob = [1.0, 1.0, 1.0, 1.0]
                for index in range(self.lam):
                    if neighbor_fitness[index] != 0:

                        cost_diff = neighbor_fitness[index] - self.pop[candidate_number + index].eval

                        # if the new solution is better, accept it
                        if cost_diff >= 0:
                            self.pop[candidate_number + index] = neighbors[index]
                        # if the new solution is not better, accept it with a probability of e^(-cost/K)
                        else:
                            r = random.uniform(0, 1)
                            ac = math.exp(cost_diff / (maxIter / 10))
                            prob[index] = ac
                            print('Cost difference : {}'.format(cost_diff))
                            print("{} < {}".format(r, ac))
                            if r < ac:
                                self.pop[candidate_number + index] = neighbors[index]
                    else:
                        eval_num -= 1

                # display and save log
                print(self._log_state(log_dir, eval_num, index=candidate_number, current_iter=current_iter,
                                      neighborFitness=neighbor_fitness, timeElapsed=process_time, probability=prob))

                # save the current population
                self.pickle_state_shc(log_dir + '/sa_files_netlists', candidate_number, maxIter, consider_neutral,
                                      log_dir, mutation_rate, eval_num, current_iter=current_iter)
