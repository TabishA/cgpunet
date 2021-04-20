#som_cosine2.py
from cgp import *
from scipy.spatial import distance
from mpl_toolkits.axes_grid1 import make_axes_locatable

def neighbourhood_function(d, t, t_max):
    sigma_0 = 20
    sigma_t = sigma_0*math.exp(-(t*math.log(sigma_0)/t_max))
    return math.exp(-(d**2/(2*sigma_t**2)))

def learning_rate(t, t_max):
    return 0.1*math.exp(-t/t_max)


def cosine_similarity_som(v1, v2):
    max1 = np.max(v1, axis=0)
    max2 = np.max(v2, axis=0)

    assert(len(max1) == len(max2))

    for i in range(len(max1)):
        m = max(max1[i], max2[i])
        for v in v1:
            if m!=0: v[i] = v[i]/m
        for v in v2:
            if m!=0: v[i] = v[i]/m
    

    similarity = np.zeros((len(v1), len(v2)))

    for i in range(similarity.shape[0]):
        for j in range(similarity.shape[1]):
            c_sim = np.dot(v1[i], v2[j])/(np.linalg.norm(v1[i])*np.linalg.norm(v2[j]))
            similarity[i][j] = c_sim

    
    a = 0 if len(v2) > len(v1) else 1

    max_sim = np.max(similarity, axis=a)
    sum_max_sim = np.sum(max_sim) - abs(len(v1) - len(v2))

    norm_sum = sum_max_sim/max(len(v1), len(v2))

    return norm_sum

class Node():
    def __init__(self, coords, weights, vectors):
        self.coords = coords
        self.weights = weights
        self.vectors = vectors
        self.neighbours = 0
        self.fitness_scores = []

    def dist(self, other_node):
        return distance.euclidean(self.coords, other_node.coords)
    
    def adapt_weights(self, input_weights, winner_node, t, t_max):
        #placeholder adaptation
        self.weights = self.weights + learning_rate(t, t_max)*neighbourhood_function(self.dist(winner_node), t, t_max)*(input_weights - self.weights)

        self.vectors = []
        for i in np.arange(0, len(self.weights), 4):
            self.vectors.append(self.weights[i:i+4])


class SOM(object):
    def __init__(self, init_population, network_info, map_size=(10,10), input_size=(128,128)):
        self.map_size = map_size
        self.init_population = init_population
        self.network_info = network_info
        self.node_list = []
        self.current_population = []
        self.converged = False
        self.map = np.zeros(self.map_size)
        self.fitness_map = np.zeros(self.map_size)
        self.input_size = input_size
        self.init_weights()
        
        
    def init_weights(self):
        self.node_list = []
        for i, ind in enumerate(self.init_population):
            sub_net_list = [[0,0,0,0] for _ in range(self.network_info.max_active_num)]
            active_net_list = ind.active_net_list()
            
            G = cgp_2_dag(active_net_list)
            vecs = dag_2_vec(G, input_size=self.input_size)

            sub_net_list[:len(vecs)] = vecs

            coord_x = int(i/self.map_size[0])
            coord_y = i%self.map_size[1]

            flat_node_weights = np.array([item for elem in sub_net_list for item in elem])

            self.node_list.append(Node((coord_x,coord_y), flat_node_weights, vecs))

    def best_matching_unit(self, input_vec):
        max_sim = 0
        bmu = self.node_list[0]
        for n in self.node_list:
            #flat_node_weights = [item for elem in n.weights for item in elem]
            n_sim = cosine_similarity_som(v1 = n.vectors, v2 =input_vec)
            if n_sim > max_sim:
                bmu = n
                max_sim = n_sim
        
        return bmu
    
    def fit(self, population, t_max):
        self.converged = False
        self.init_weights()

        for t in range(t_max):

            for n in self.node_list:
                n.neighbours = 0
                n.fitness_scores = []

            for ind in population:
                sub_net_list = [[0,0,0,0] for _ in range(self.network_info.max_active_num)]
                active_net_list = ind.active_net_list()
            
                G = cgp_2_dag(active_net_list)
                vecs = dag_2_vec(G, input_size=self.input_size)

                sub_net_list[:len(vecs)] = vecs
                flat_net_list = np.array([item for elem in sub_net_list for item in elem])

                bmu = self.best_matching_unit(vecs)
                bmu.neighbours = bmu.neighbours + 1
                bmu.fitness_scores.append(ind.eval)
            
                for node in self.node_list:
                    node.adapt_weights(flat_net_list, bmu, t, t_max)
        

        self.converged = True

    
    def draw(self, save_dir=None):
        self.map = np.zeros(self.map_size)
        self.fitness_map = np.zeros(self.map_size)
        
        for n in self.node_list:
            self.map[n.coords] = n.neighbours
            if n.fitness_scores:
                self.fitness_map[n.coords] = int(np.mean(n.fitness_scores)*100)
        
        fig, axs = plt.subplots(1,2, sharey=True)
        im1 = axs[0].imshow(self.map.astype(int), cmap='Reds', label='Neighbourhood')
        im2 = axs[1].imshow(self.fitness_map.astype(int), cmap='Reds', label='Fitness Map')

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im2, cax=cax, orientation='vertical')

        if save_dir:
            plt.savefig(save_dir)
        else:
            plt.show()
        
        plt.close()