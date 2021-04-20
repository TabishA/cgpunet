import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import sys
import re

def check_func(func_gene, func):
    elems = func_gene.split('_')
    if elems[0] == 'S':
        return True if elems[1] == func else False
    else:
        return True if elems[0] == func or elems[0] + '_' + elems[1] == func else False


#NOTE: "successors" should be renamed to predecessors
def get_function(node):
    func_dict = dict()
    
    elems = node.split('_')
    
    if elems[0] == 'S' or elems[0] == 'D':
        func_dict['function'] = elems[1]
        func_dict['filters'] = int(elems[2])
        func_dict['kernel_size'] = (int(elems[3]), int(elems[3]))
        func_dict['successors'] = (int(elems[4]), int(elems[5]))
    elif elems[0] == 'Max' or elems[0] == 'Avg':
        func_dict['function'] = elems[0] + '_' + elems[1]
        func_dict['successors'] = (int(elems[2]), int(elems[3]))
    elif (re.search("^Sum.*", elems[0]) or re.search("^Concat.*", elems[0]) or re.search("^input.*", elems[0])):
        func_dict['function'] = elems[0]
        func_dict['successors'] = (int(elems[1]), int(elems[2]))
    elif re.search("^FC.*", elems[0]):
        func_dict['function'] = elems[0]
        func_dict['units'] = int(elems[1])
        func_dict['successors'] = (int(elems[2]), int(elems[2]))
    
    return func_dict



def get_data(G, incoming_node, field):
    elems = incoming_node.split('_')
    if (re.search("^ConvBlock.*", elems[1]) or re.search("^ResBlock.*", elems[1]) or re.search("^DeconvBlock.*", elems[1])) and field == 'num_channels':
        return elems[2]
    else:
        data = list(G.in_edges(incoming_node, data=True))
        #print('getting data for node {}: {}'.format(incoming_node, data))
        return data[0][2][field]



def merge_nodes(G, node1, node2, mode='concat'):
    print('MERGING NODES')

    node1_id = int(G.nodes[node1]['id'])
    node2_id = int(G.nodes[node2]['id'])
    node1_ch = int(G.nodes[node1]['num_channels'])
    node2_ch = int(G.nodes[node2]['num_channels'])
    node1_pool = int(G.nodes[node1]['pool_factor'])
    node2_pool = int(G.nodes[node2]['pool_factor'])

    new_node_id = max(node1_id, node2_id) + 1
    pool_factor = max(node1_pool, node2_pool) 

    if mode == 'sum':
        new_node = 'Sum_' + str(node1_id) + '_' + str(node2_id)
        num_channels = max(node1_ch, node2_ch)
    elif mode == 'concat':
        new_node = 'Concat_' + str(node1_id) + '_' + str(node2_id)
        num_channels = node1_ch + node2_ch
    else:
        sys.exit('unrecognized mode in merge_nodes')
        
    
    G.add_node(new_node, num_channels=num_channels, pool_factor=pool_factor, id=new_node_id)
    G.add_edge(node1, new_node, num_channels = node1_ch, pool_factor = node1_pool)
    G.add_edge(node2, new_node, num_channels = node2_ch, pool_factor = node2_pool)

    return G


def get_leaves(G):
    G_nodes = G.nodes()
    leaves = [x for x in G_nodes if G.out_degree(x)==0 and G.in_degree(x)>0]
    return leaves


#appending output layers to obtain desired output size
def append_output_layers(G, fc_layers, num_classes = 10):
    G_nodes = G.nodes()
    leaves = get_leaves(G)
    
    while len(leaves) > 1:
        G = merge_nodes(G, leaves[0], leaves[1], mode='concat')
        leaves = get_leaves(G)
      
    if len(leaves) < 1: sys.exit('no leaves - cyclic graph: {}'.format(len(leaves)))  
    
    leaf = leaves[0]
    leaf_id = int(G.nodes[leaf]['id'])

    for fc in fc_layers:
        node_list = list(G.nodes())
        elems = fc.split('_')
        fc_node = fc + '_' + str(leaf_id) + '_' + str(leaf_id)
        G.add_node(fc_node, units=int(elems[1]), num_channels=0, pool_factor=0, id=leaf_id+1)
        G.add_edge(node_list[leaf_id], fc_node, num_channels=0, pool_factor=0)
        leaf_id += 1
    
    

    leaves = get_leaves(G)
    leaf = leaves[0]
    leaf_id = int(G.nodes[leaf]['id'])

    #Final output layer
    output_node = 'FC_' + str(num_classes) + '_' + str(leaf_id) + '_' + str(leaf_id)
    G.add_node(output_node, units=num_classes, num_channels=0, pool_factor=0, id=leaf_id+1)
    node_list = list(G.nodes())
    G.add_edge(node_list[leaf_id], output_node, num_channels=0, pool_factor=0)


        
    return G
    


#this function generates an index for duplicate nodes since the networkx Graph object will not allow duplicate node names
def increment_duplicate_index(elem, i):
    sub_elems = elem.split('_')
    if len(sub_elems) == 1:
        return elem + str(i)
    else:
        out = sub_elems[0]
        for j in range(1, len(sub_elems)):
            out = out + '_' + sub_elems[j]
            if j == 1: out = out + str(i)
        return out


#TODO: input shape as args
def cgp_2_dag(net_list, fc_layers):
    print('Individual: {}'.format(net_list))

    G = nx.MultiDiGraph()

    for elem in net_list:
        node = elem[0] + '_' + str(elem[1]) + '_' + str(elem[2])
        if node not in list(G.nodes()):
            G.add_node(node)
        else:
            i = 0
            func_name = increment_duplicate_index(elem[0], i)
            node = func_name + '_' + str(elem[1]) + '_' + str(elem[2])
            while node in list(G.nodes()):
                i += 1
                func_name = increment_duplicate_index(elem[0], i)
                node = func_name + '_' + str(elem[1]) + '_' + str(elem[2])
            G.add_node(node)

    node_list = list(G.nodes())

    

    pool_factor = 0
    init_num_channels = 1
    num_channels = 1

    for i in range(len(node_list)):
        node = node_list[i]
        
        #last 2 elements will be connection genes
        sub_elements = node.split('_')
        op = sub_elements[0]
        
        if not (re.search("^Sum.*", op) or re.search("^Concat.*", op)): #single input functions
            connect_index = int(sub_elements[len(sub_elements) - 2])
            in_node = node_list[connect_index]

            print('node: {}, in_node: {}'.format(node, in_node))
            
            if op == 'input':
                #print('HELLO INPUT NODE')
                G.add_node(node, num_channels=init_num_channels, pool_factor=0, id=i)
                pool_factor = 0
                num_channels = 1
            elif (re.search("^ConvBlock.*", sub_elements[1])):
                num_channels = get_data(G, node, 'num_channels')
                pool_factor = G.nodes[in_node]['pool_factor']
                G.add_node(node, num_channels=num_channels, pool_factor=pool_factor, id=i)
            elif (re.search("^ResBlock.*", sub_elements[1])):
                num_channels = get_data(G, node, 'num_channels')
                in_ch = G.nodes[in_node]['num_channels']
                pool_factor = G.nodes[in_node]['pool_factor']
                G.add_node(node, num_channels= max(int(num_channels), int(in_ch)), pool_factor=pool_factor, id=i)
            elif (re.search("^DeconvBlock.*", sub_elements[1])):
                num_channels = get_data(G, node, 'num_channels')
                pool_factor = G.nodes[in_node]['pool_factor'] - 1
                G.add_node(node, num_channels=num_channels, pool_factor=pool_factor, id=i)
            elif op == 'Max' or op == 'Avg': # Max_Pool or Avg_Pool
                num_channels = G.nodes[in_node]['num_channels']
                pool_factor = G.nodes[in_node]['pool_factor'] + 1
                G.add_node(node, num_channels=num_channels, pool_factor=pool_factor, id=i)

            if in_node != node:
                in_ch = G.nodes[in_node]['num_channels']
                in_pool = G.nodes[in_node]['pool_factor']
                G.add_edge(in_node, node, num_channels = in_ch, pool_factor = in_pool)
                #print('in_ch: {}, in_pool:{}'.format(in_ch, in_pool))
           
                # node_ch = get_data(G, node, 'num_channels')
                # node_pool = get_data(G, node, 'pool_factor')
                #print('node_ch: {}, node_pool:{}'.format(node_ch, node_pool))
        else:
            connect_index_1 = int(sub_elements[len(sub_elements) - 1])
            connect_index_2 = int(sub_elements[len(sub_elements) - 2])

            in_node_1 = node_list[connect_index_1]
            in_node_2 = node_list[connect_index_2]
        
            node1_ch = int(G.nodes[in_node_1]['num_channels'])
            node2_ch = int(G.nodes[in_node_2]['num_channels'])

            node1_pool = int(G.nodes[in_node_1]['pool_factor'])
            node2_pool = int(G.nodes[in_node_2]['pool_factor'])

            #when combining feature maps of different sizes, we apply max pooling to the larger feature map to match size
            pool_factor = max(node1_pool, node2_pool) 

            if re.search("^Sum.*", op):
                #when adding, we match the number of channels in both feature maps by zero padding
                num_channels = max(node1_ch, node2_ch)
            elif re.search("^Concat.*", op):
                #feature maps are concatenated along the channels axis, so add the number of channels
                num_channels = node1_ch + node2_ch
        
            G.add_node(node, num_channels=num_channels, pool_factor=pool_factor, id=i)
        
            if in_node_1 != node:
                G.add_edge(in_node_1, node, num_channels = node1_ch, pool_factor = node1_pool)
       
            if in_node_2 != node:
                G.add_edge(in_node_2, node, num_channels = node2_ch, pool_factor = node2_pool)

    G = append_output_layers(G, fc_layers)
    
    return G


def draw_dag(G, save_to_dir=None):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    edge_labels = nx.get_edge_attributes(G,'num_channels')
    nx.draw_networkx_edge_labels(G, pos, labels = edge_labels, font_size=5)
    
    if save_to_dir is None:
        plt.show()
    else:
        plt.figure()
        plt.savefig(save_to_dir)

