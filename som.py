#som.py
from cgp import *
from scipy.spatial import distance
from mpl_toolkits.axes_grid1 import make_axes_locatable

def neighbourhood_function(d, t, t_max):
    sigma_0 = 20
    sigma_t = sigma_0*math.exp(-(t*math.log(sigma_0)/t_max))
    return math.exp(-(d**2/(2*sigma_t**2)))

def learning_rate(t, t_max):
    return 0.1*math.exp(-t/t_max)

class Node():
    def __init__(self, coords, weights):
        self.coords = coords
        self.weights = weights
        self.neighbours = 0
        self.fitness_scores = []

    def dist(self, other_node):
        return distance.euclidean(self.coords, other_node.coords)
    
    def adapt_weights(self, input_weights, winner_node, t, t_max):
        #placeholder adaptation
        self.weights = self.weights + learning_rate(t, t_max)*neighbourhood_function(self.dist(winner_node), t, t_max)*(input_weights - self.weights)


class SOM(object):
    def __init__(self, init_population, network_info, map_size=(10,10)):
        self.map_size = map_size
        self.init_population = init_population
        self.network_info = network_info
        self.node_list = []
        self.current_population = []
        self.converged = False
        self.map = np.zeros(self.map_size)
        self.fitness_map = np.zeros(self.map_size)
        self.init_weights()
    
    def init_weights(self):
        self.node_list = []
        for i, ind in enumerate(self.init_population):
            sub_net_list = [[0,0,0] for _ in range(self.network_info.max_active_num)]
            active_net_list = ind.active_net_list()
            
            for n in active_net_list:
                if n[0] == 'input': continue
                n[0] = self.network_info.func_type.index(n[0])
            
            sub_net_list[:len(active_net_list)-1] = active_net_list[1:]
            coord_x = int(i/self.map_size[0])
            coord_y = i%self.map_size[1]

            flat_node_weights = np.array([item for elem in sub_net_list for item in elem])

            self.node_list.append(Node((coord_x,coord_y), flat_node_weights))
    

    def best_matching_unit(self, flat_net_list):
        min_dist = 1e7
        bmu = self.node_list[0]
        for n in self.node_list:
            #flat_node_weights = [item for elem in n.weights for item in elem]
            n_dist = distance.euclidean(flat_net_list, n.weights)
            if n_dist < min_dist:
                bmu = n
                min_dist = n_dist
        
        return bmu
    
    def fit(self, population, t_max):
        self.converged = False
        self.init_weights()

        for t in range(t_max):

            for n in self.node_list:
                n.neighbours = 0
                n.fitness_scores = []

            for ind in population:
                sub_net_list = [[0,0,0] for _ in range(self.network_info.max_active_num)]
                active_net_list = ind.active_net_list()
                for n in active_net_list:
                    if n[0] == 'input': continue
                    n[0] = self.network_info.func_type.index(n[0])
        
                sub_net_list[:len(active_net_list)-1] = active_net_list[1:]

                #flatten list of lists
                flat_net_list = np.array([item for elem in sub_net_list for item in elem])

                bmu = self.best_matching_unit(flat_net_list)
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