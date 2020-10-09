#dag_2_cnn.py

from cgp_2_dag import *
import tensorflow as tf
from blocks import *
#from keras.models import *
from tensorflow.keras.models import *
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as metrics
import re
import os

tf.random.set_seed(1)

def traverse(dag, successor, modules):
    preds = list(dag.predecessors(successor))
    
    func_dict = get_function(successor)
    function = func_dict['function']

    m_keys = list(modules.keys())

    if successor in m_keys: return modules
    
    #print('modules: {}'.format(modules))
    #  if successor in list(modules.keys()):
    #      print('{} already in keys'.format(successor))

    if len(preds) == 1:
        
        if not preds[0] in m_keys:
            print('preds[0] not in keys: {}'.format(preds[0])) 
            modules = traverse(dag, preds[0], modules)
        
        if re.search("^ConvBlock.*", function):
            f = func_dict['filters']
            k = func_dict['kernel_size']
            modules[successor] = ConvBlock(f, kernel_size=k, conv_name=successor)(modules[preds[0]])
            print('connecting {} to {}'.format(modules[successor], modules[preds[0]]))
        elif re.search("^ResBlock.*", function):
            f = func_dict['filters']
            k = func_dict['kernel_size']
            modules[successor] = ResBlock(modules[preds[0]], f, kernel_size=k, res_name=successor)(modules[preds[0]])
            print('connecting {} to {}'.format(modules[successor], modules[preds[0]]))
        elif re.search("^DeconvBlock.*", function):
            f = func_dict['filters']
            k = func_dict['kernel_size']
            modules[successor] = DeconvBlock(modules[preds[0]], f, kernel_size=k, deconv_name=successor)(modules[preds[0]])
            print('connecting {} to {}'.format(modules[successor], modules[preds[0]]))
        elif re.search("^Max_Pool.*", function):
            modules[successor] = MaxPooling2D(pool_size=(2, 2), name=successor)(modules[preds[0]])
            print('connecting {} to {}'.format(modules[successor], modules[preds[0]]))
        elif re.search("^Avg_Pool.*", function):
            modules[successor] = AveragePooling2D(pool_size=(2, 2), name=successor)(modules[preds[0]])
            print('connecting {} to {}'.format(modules[successor], modules[preds[0]]))
        elif re.search("^Sum.*", function):
            modules[successor] = MergeBlock(modules[preds[0]], modules[preds[0]], mode='add', block_name=successor)
            print('connecting {} to {} and {}'.format(modules[successor], modules[preds[0]], modules[preds[0]]))
        elif re.search("^Concat.*", function):
            modules[successor] = MergeBlock(modules[preds[0]], modules[preds[0]], mode='concat', block_name=successor)
            print('connecting {} to {} and {}'.format(modules[successor], modules[preds[0]], modules[preds[0]]))
    elif len(preds) == 2:
        if not preds[0] in m_keys: 
            modules = traverse(dag, preds[0], modules)

        if not preds[1] in m_keys: 
            modules = traverse(dag, preds[1], modules)
        
        pred_1 = modules[preds[0]]
        pred_2 = modules[preds[1]]

        if re.search("^Sum.*", function):
            modules[successor] = MergeBlock(pred_1, pred_2, mode='add', block_name=successor)
            print('connecting {} to {} and {}'.format(modules[successor], modules[preds[0]], modules[preds[1]]))
        elif re.search("^Concat.*", function):
            modules[successor] = MergeBlock(pred_1, pred_2, mode='concat', block_name=successor)
            print('connecting {} to {} and {}'.format(modules[successor], modules[preds[0]], modules[preds[1]]))

    return modules


def dag_2_cnn(dag, gpuID, input_shape=(256,256,1), target_shape=(256,256,1), pretrained_weights = None, compile=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)
    nodes = list(dag.nodes())
    
    #breadth-first search starting at root
    bfs = nx.bfs_successors(dag, nodes[0])
    modules = dict()

    #root will always have this name
    assert(nodes[0] == 'input_0_0')

    with tf.device('/gpu:{}'.format(gpuID)):
        modules[nodes[0]] = Input(input_shape)
        
        for branch in bfs: #branch: tuple with (node, [list of successors to node])
            for successor in branch[1]:
                modules = traverse(dag, successor, modules)
                
        leaves = [x for x in dag.nodes() if dag.out_degree(x)==0 and dag.in_degree(x)>0]
        
        if len(leaves) == 1:
            output = modules[leaves[0]]
        else:
            raise NotImplementedError
        
        #NOTE: mean iou removed from metrics 21/07/2020
        model = Model(inputs=modules['input_0_0'], outputs=output)
        if compile:
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.TruePositives(), metrics.TrueNegatives(), metrics.FalsePositives(), metrics.FalseNegatives(), metrics.AUC()])
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        
        if pretrained_weights:
            model.load_weights(pretrained_weights)

    return model
