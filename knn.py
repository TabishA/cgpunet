import numpy as np
import networkx as nx

# Locate the most similar neighbors
def get_neighbours(pop, ind, num_neighbors, input_size):
	distances = list()
	for p in pop:
		dist = cosine_similarity(ind, p, input_size)
		distances.append((p, dist))
	distances.sort(key=lambda tup: tup[1], reverse=True)
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i])
	return neighbors


def dag_2_vec(G, input_size = (128,128)):
    vec = []
    depth = nx.shortest_path_length(G, 'input_0_0')
    for node in G.nodes():
        node_dict = G.nodes[node]
        pf = node_dict['pool_factor']
        num_ch = node_dict['num_channels']
        if pf > 0:
            x = input_size[0]*(2*pf)
            y = input_size[1]*(2*pf)
        elif pf == 0 and num_ch == 0:
            x = 1
            y = int(node_dict['units'])
        elif pf == 0:
            x = input_size[0]
            y = input_size[1]
        else:
            x = input_size[0]/(2*abs(pf))
            y = input_size[1]/(2*abs(pf))
        vec.append([int(node_dict['num_channels']), x, y, depth[node]])

    return vec


def cosine_similarity(G1, G2, input_size):
    v1 = dag_2_vec(G1, input_size)
    v2 = dag_2_vec(G2, input_size)

    v1 = v1[1:len(v1)-1]
    v2 = v2[1:len(v2)-1]

    #in case of unequal length, make sure v1 is the longer of the two
    if len(v2) > len(v1):
        temp = v1.copy()
        v1 = v2
        v2 = temp
    
    length_diff = len(v1) - len(v2)

    for _ in range(length_diff):
        v2.append([0, 0, 0, 0])

    assert(len(v1) == len(v2))
    
    similarity = np.zeros(len(v1))

    for i in range(len(v1)):
        if np.linalg.norm(v2[i]) == 0:
            c_sim = 0
        else:
            c_sim = np.dot(v1[i], v2[i])/(np.linalg.norm(v1[i])*np.linalg.norm(v2[i]))
        
        similarity[i] = c_sim

    
    sum_sim = np.sum(similarity)

    norm_sum = sum_sim/len(v1)

    return norm_sum
