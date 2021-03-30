from datetime import datetime

from cgp import *
import numpy as np
import math
from numpy.linalg import norm
import seaborn as sb
import matplotlib.pyplot as plt

# Hyperparameters

act_thresh = 0.05
epsilon = 0.0001  # constant to avoid divide by 0 error
m = 1  # Dimensions of the input vector
c = 0.5  # Connection threshold
s = 0.05  # Slope of logistic function
eb = 0.001  # Learning rate
beta = 0.001
en = 0.0001 * eb  # Learning rate
dir_name = os.path.join(os.getcwd(), 'SOM_' + datetime.now().strftime('%m-%d-%H-%M'))


class Node():
    def __init__(self, weights):
        self.weights = weights
        self.relevence = np.ones(len(weights))  # defines importance of the ith weight param
        self.avgDistance = np.zeros(len(weights))  # used to update the relevence vector
        self.activations = []
        self.wins = 1
        self.testactivations = []
        self.testwins = 1
        self.neighbors = []


class SOM(object):
    def __init__(self, init_population, network_info, N):
        self.max_nodes = N
        self.init_population = init_population
        self.network_info = network_info
        self.node_list = []
        self.train()

    def train(self):
        # Define hyperparameters
        print("Training SOM")
        for i, ind in enumerate(self.init_population):
            active_net_list = ind.active_net_list()
            for n in active_net_list:
                if n[0] == 'input':
                    n[0] = -1
                else:
                    n[0] = self.network_info.func_type.index(n[0])

            input_weights = np.array([item for elem in active_net_list for item in elem])

            # If there is no node in the map
            if len(self.node_list) == 0:
                self.node_list.append(Node(input_weights))


            else:
                winner, activation = self.compute_activation(input_weights)

                if activation < act_thresh and len(self.node_list) < self.max_nodes:
                    # Create a new node and connect it
                    winner = Node(input_weights)
                    winner.activations.append(activation)
                    self.node_list.append(winner)
                    self.update_neighbors(winner)
                else:
                    # Update the winner and its neighbors
                    winner.activations.append(activation)
                    self.update_winner_parameters(winner, input_weights)
                    self.update_neighbor_parameters(winner, input_weights)
                    winner.wins += 1
        self.save_params()

    def compute_activation(self, input_weights):
        max = 0
        winner = Node([])
        for node in self.node_list:
            if len(node.weights) == len(input_weights):
                d = self.get_weighted_distance(node, input_weights, 0, len(input_weights))
                r_sum = sum(node.relevence)
                activation = r_sum / (r_sum + d + epsilon)
                if activation > max:
                    max = activation
                    winner = node

            elif len(node.weights) > len(input_weights):
                len_diff = len(node.weights) - len(input_weights) + 1
                for iter in range(len_diff):
                    d = self.get_weighted_distance(node, input_weights, iter, len(input_weights) + iter)
                    r_sum = sum(node.relevence[iter:len(input_weights) + iter])
                    activation = r_sum / (r_sum + d + epsilon)
                    if activation > max:
                        max = activation
                        winner = node

            else:
                d = self.get_weighted_distance(node, input_weights, 0, len(node.weights))
                r_sum = sum(node.relevence)
                activation = r_sum / (r_sum + d + epsilon)
                if activation > max:
                    max = activation
                    winner = node

        if len(winner.weights) < len(input_weights):
            # Update the node vectors
            for i in range(len(winner.weights), len(input_weights)):
                winner.weights = np.append(winner.weights, input_weights[i])
                winner.relevence = np.append(winner.relevence, 0.5)
                winner.avgDistance = np.append(winner.avgDistance, 0)
        return winner, activation

    def get_weighted_distance(self, node, input_weights, start_index=0, end_index=0):
        weight = node.weights
        relevence = node.relevence
        if end_index == 0:
            end_index = len(input_weights)
        sum = 0
        for i in range(start_index, end_index):
            sum += relevence[i] * pow(input_weights[i - start_index] - weight[i], 2)
        return math.sqrt(sum)

    def update_neighbors(self, winner):
        for node in self.node_list:
            if node != winner:
                w1 = winner.relevence
                w2 = node.relevence
                # Check the length mis match
                if len(w1) < len(w2):
                    w1 = np.append(w1, [0] * (len(w2) - len(w1)))
                elif len(w1) > len(w2):
                    w2 = np.append(w2, [0] * (len(w1) - len(w2)))
                arr = [x1 - x2 for (x1, x2) in zip(w1, w2)]

                if norm(arr, 1) < c * math.sqrt(m):
                    winner.neighbors.append(node)

    def update_winner_parameters(self, winner, input_weights):

        index = min(len(winner.weights), len(input_weights))
        # update weights
        for i in range(index):
            winner.weights[i] = winner.weights[i] + eb * (input_weights[i] - winner.weights[i])

        # update avg distance
        for i in range(index):
            winner.avgDistance[i] = (1 - eb * beta) * winner.avgDistance[i] + \
                                    (eb * beta * math.fabs((input_weights[i] - winner.weights[i])))

        # update relevence
        dmin = min(winner.avgDistance)
        dmax = max(winner.avgDistance)
        davg = sum(winner.avgDistance) / len(winner.avgDistance)
        for i in range(index):
            if dmin != dmax:
                x = (davg - winner.avgDistance[i]) / (s * (dmax - dmin))
                winner.relevence[i] = 1 / (1 + math.exp(x))
            else:
                winner.relevence[i] = 1

    def update_neighbor_parameters(self, winner, input_weights):

        for node in winner.neighbors:
            index = min(len(node.weights), len(input_weights))
            # update weights
            for i in range(index):
                node.weights[i] = node.weights[i] + en * (input_weights[i] - node.weights[i])

            # update avg distance
            for i in range(index):
                node.avgDistance[i] = (1 - en * beta) * node.avgDistance[i] + \
                                      (en * beta * math.fabs((input_weights[i] - node.weights[i])))

            # update relevence
            dmin = min(node.avgDistance)
            dmax = max(node.avgDistance)
            davg = sum(node.avgDistance) / len(node.avgDistance)
            for i in range(index):
                if dmin != dmax:
                    x = (davg - node.avgDistance[i]) / (s * (dmax - dmin))
                    node.relevence[i] = 1 / (1 + math.exp(x))
                else:
                    node.relevence[i] = 1

    def draw_map(self, data, rows=5, title='', filename='sample.png'):
        mask = np.zeros(data.shape)
        cols = data.shape[0] // rows
        diff = 0
        if data.shape[0] % rows != 0:
            cols += 1
            diff = cols * rows - data.shape[0]
        data = np.append(data, [-1] * diff)
        mask = np.append(mask, [1] * diff)

        data = data.reshape(rows, cols)
        mask = mask.reshape(rows, cols)

        fig, ax = plt.subplots(1)
        heatmap = sb.heatmap(data, annot=True, mask=mask, xticklabels=False, yticklabels=False)
        heatmap.set_title(title)
        heatmap.get_figure().savefig(filename)

    def save_params(self):
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError as e:
                print(e)
                return
        params = {'Log Directory': dir_name,
                  'Population Length': len(self.init_population),
                  'Activation Threshold': act_thresh,
                  'Epsilon': epsilon,
                  'Input Dimensions': m,
                  'Connection Threshold': c,
                  'Logistic Slope': s,
                  'Learning Rate Winner': eb,
                  'Learning Rate Neighbor': en,
                  'Beta': beta}
        pickle.dump(params, open(os.path.join(dir_name, 'param.pickle'), 'wb'))
        pickle.dump(self, open(os.path.join(dir_name, 'som.pickle'), 'wb'))
        pickle.dump(self.init_population, open(os.path.join(dir_name, 'trainpop.pickle'), 'wb'))

        wins = []
        avg_activation = []
        for node in self.node_list:
            wins.append(node.wins)
            if len(node.activations) > 0:
                avg_activation.append(sum(node.activations) / len(node.activations))
            else:
                avg_activation.append(0)
        self.draw_map(np.array(wins), title='Heatmap of number of wins of nodes in SOM',
                      filename=os.path.join(dir_name, 'wins.png'))
        self.draw_map(np.array(avg_activation), title='Heatmap of avergae activation of nodes in SOM',
                      filename=os.path.join(dir_name, 'act.png'))

    def fit(self, population, generation=0):
        clusters = np.zeros(len(self.node_list))
        outliers = 0
        for i, ind in enumerate(population):
            active_net_list = ind.active_net_list()
            for n in active_net_list:
                if n[0] == 'input':
                    n[0] = -1
                else:
                    n[0] = self.network_info.func_type.index(n[0])

            input_weights = np.array([item for elem in active_net_list for item in elem])

            winner, activation = self.compute_activation(input_weights)

            if activation >= act_thresh:
                # Assign the input to the cluster of the winning node
                clusters[self.node_list.index(winner)] += 1
            else:
                print(activation)
                outliers += 1

        print('Total Number of clusters for a population of length {} is {}'.format(len(population),
                                                                                    np.count_nonzero(clusters)))
        print('Number of outliers are {}'.format(outliers))
        print('Percentage of samples clustered {}'.format((len(population) - outliers) * 100 / len(population)))
        print('Percentage of nodes activated {}'.format((np.count_nonzero(clusters)) * 100
                                                        / len(self.node_list)))

        if not os.path.isdir(os.path.join(dir_name, 'fit')):
            try:
                os.makedirs(os.path.join(dir_name, 'fit'))
            except OSError as e:
                print(e)
                return
        self.draw_map(clusters, title='Heatmap of fitting test population {} to SOM'.format(generation),
                      filename=os.path.join(dir_name, 'fit', 'test{}.png'.format(generation)))

    def map_population(self, population, gen):
        print("Mapping population {} to SOM".format(gen))
        self.fit(population, gen)
        for i, ind in enumerate(population):
            active_net_list = ind.active_net_list()
            for n in active_net_list:
                if n[0] == 'input':
                    n[0] = -1
                else:
                    n[0] = self.network_info.func_type.index(n[0])

            input_weights = np.array([item for elem in active_net_list for item in elem])

            winner, activation = self.compute_activation(input_weights)

            if activation >= act_thresh:
                winner.testactivations.append(activation)
                winner.testwins += 1
                self.update_winner_parameters(winner, input_weights)
                self.update_neighbor_parameters(winner, input_weights)

        if not os.path.isdir(os.path.join(dir_name, 'map')):
            try:
                os.makedirs(os.path.join(dir_name, 'map'))
            except OSError as e:
                print(e)
                return
        wins = [node.testwins for node in self.node_list]
        self.draw_map(np.array(wins), title='Heatmap of mapping test population {} to SOM'.format(gen),
                      filename=os.path.join(dir_name, 'map', 'test{}.png'.format(gen)))
