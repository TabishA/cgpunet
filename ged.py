from dataHelperTif import get_files
from cgp_2_dag import cgp_2_dag, draw_dag, get_function
import os
import pickle
from networkx.algorithms import similarity
import concurrent.futures
import time


def match(n1, n2):
    # return n1['num_channels'] == n2['num_channels'] and n1['pool_factor'] == n2['pool_factor']
    return n1['function'] == n2['function']


def ged(args):
    timeout = 300

    ind1, ind2 = args
    G1 = cgp_2_dag(ind1.active_net_list())
    G2 = cgp_2_dag(ind2.active_net_list())
    # G1 = ind1
    # G2 = ind2
    duration = 0
    start = time.time()
    sim_gp = None
    while duration <= timeout and sim_gp == None:
        for v in similarity.optimize_graph_edit_distance(G1, G2, node_match=match):
            sim_gp = v
            duration = time.time() - start
            if duration > timeout:
                print('Timeout')
                break
        break

    print('SIM: {}'.format(sim_gp))

    return sim_gp, ind2


def get_iterable(pop, ind):
    it = []
    for p in pop:
        it.append((ind, p))
    return it


# Locate the most similar neighbors
def get_neighbours(pop, ind, num_neighbors):
    distances = list()
    args = get_iterable(pop, ind)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(ged, args)

    for dist, ind2 in result:
        distances.append((ind2, dist))

    # distances.sort(key=lambda tup: tup[1], reverse=True)
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i])

    return neighbors


def get_distances(pop, ind):
    distances = dict()
    args = get_iterable(pop, ind)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        result = executor.map(ged, args)

    for dist, ind2 in result:
        distances[ind2.model_name] = dist

    # distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
    # distances.sort(key=lambda tup: tup[1])

    return distances


def get_distances_simgnn(simgnn_model, data):
    """
    data : a list of dictionaries of this form [{g1, g2, l1, l2, m1, m2}]

    return : {individual.modelname: ged}
    """
    print('get distances Sim GNN')
    result = dict()
    for pair in data:
        print('pair: {}'.format(pair))
        result[pair['modelname2']] = simgnn_model.predict(pair)

# New get distance function to be coded here using the trained model
# Model inputs are individuals in json format
def entropy(pop):
    pass
