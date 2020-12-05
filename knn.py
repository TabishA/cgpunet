import numpy as np
import networkx as nx


def calculate_distance(pop, ind):
    distances = list()
    for p in pop:
        dist = cosine_similarity(ind, p)
        distances.append((p, dist))
    distances.sort(key=lambda tup: tup[1], reverse=True)
    return distances


# Function to check if a neighboring solution is in the local neighborhood of the candidate solution
def check_local_neighbor(pop, candidate, neighbor):
    distances = calculate_distance(pop, neighbor)
    max_similarity = distances[0][1]
    print("Best similarity is {} with model {}".format(max_similarity, distances[0][0]))
    for item in distances:
        if item[0] == candidate:
            print("Similarity with model {} is {}".format(candidate, item[1]))
            return item[1] == max_similarity

        # Locate the most similar neighbors


def get_neighbours(pop, ind, num_neighbors):
    distances = calculate_distance(pop, ind)
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i])
    return neighbors


def dag_2_vec(G, input_size=(128, 128)):
    vec = []
    depth = nx.shortest_path_length(G, 'input_0_0')
    for node in G.nodes():
        node_dict = G.nodes[node]
        pf = node_dict['pool_factor']
        if pf >= 0:
            x = input_size[0] * (2 * pf)
            y = input_size[1] * (2 * pf)
        else:
            x = input_size[0] / (2 * abs(pf))
            y = input_size[1] / (2 * abs(pf))
        vec.append([int(node_dict['num_channels']), x, y, depth[node]])

    return vec


def cosine_similarity(G1, G2):
    v1 = dag_2_vec(G1)
    v2 = dag_2_vec(G2)

    max1 = np.max(v1, axis=0)
    max2 = np.max(v2, axis=0)

    assert (len(max1) == len(max2))

    for i in range(len(max1)):
        m = max(max1[i], max2[i])
        for v in v1:
            if m != 0: v[i] = v[i] / m
        for v in v2:
            if m != 0: v[i] = v[i] / m

    similarity = np.zeros((len(v1), len(v2)))

    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            c_sim = np.dot(v1[i], v2[j]) / (np.linalg.norm(v1[i]) * np.linalg.norm(v2[j]))
            similarity[i][j] = c_sim

    a = 0 if len(v2) > len(v1) else 1

    max_sim = np.max(similarity, axis=a)
    sum_max_sim = np.sum(max_sim) - abs(len(v1) - len(v2))

    norm_sum = sum_max_sim / max(len(v1), len(v2))

    return norm_sum
