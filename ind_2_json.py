#ind_2_json.py
from cgpunet_main import *
import pandas as pd
import json
from sklearn.model_selection import train_test_split

labels = {'input': '0', 'ConvBlock': '1', 'ResBlock': '2', 'DeconvBlock': '3', 'Concat': '4', 'Sum': '5', \
    'Avg_Pool': '6', 'Max_Pool': '7'}

def ind_2_dict(ind):
    ind_edgelist = []
    ind_labels = []

    netlist = ind.active_net_list()
    G = cgp_2_dag(netlist)
    nodelist = list(G.nodes)

    for node in nodelist:
        elems = node.split('_')
        fn = G.nodes[node]['function']
        id = int(G.nodes[node]['id'])
        if fn == 'input':
            ind_labels.append(labels[fn])
            continue
        elif fn in ['Sum', 'Concat']:
            n1 = int(elems[1])
            n2 = int(elems[2])
            edge1 = [n1, id]
            edge2 = [n2, id]
            ind_edgelist.append(edge1)
            ind_labels.append(labels[fn])
            ind_edgelist.append(edge2)
            ind_labels.append(labels[fn])
        else:
            n1 = int(elems[len(elems)-2])
            ind_edgelist.append([n1, id])
            ind_labels.append(labels[fn])

    return {'graph': ind_edgelist, 'labels': ind_labels}


if __name__ == "__main__":
    df_ged = pd.read_pickle('df_ged_concat.p')
    ind_dict = pickle.load(open('ind_dict.p', 'rb'))

    os.makedirs('./dataset')

    for i, row in enumerate(df_ged.iterrows()):
        outname = str(i) + '.json'
        outname = os.path.join('./dataset', outname)
        ind1 = ind_dict[row[1]['model_1']]
        ind2 = ind_dict[row[1]['model_2']]
        ged = row[1]['ged']

        ind1_dict = ind_2_dict(ind1)
        ind2_dict = ind_2_dict(ind2)

        out = {"graph_1": ind1_dict['graph'], "graph_2": ind2_dict['graph'], \
             "labels_1": ind1_dict['labels'], "labels_2": ind2_dict['labels'], "ged": int(row[1]['ged'])}

        with open(outname, 'w') as outfile:
            json.dump(out, outfile)

    
    all_files = get_files('./dataset/', '*.json')
    train_path = os.path.join('./dataset', 'train')
    test_path = os.path.join('./dataset', 'test')

    os.makedirs(train_path)
    os.makedirs(test_path)

    train, test = train_test_split(all_files, test_size=0.2)

    for f in train:
        fname = os.path.basename(f)
        os.replace(f, os.path.join(train_path, fname))

    for f in test:
        fname = os.path.basename(f)
        os.replace(f, os.path.join(test_path, fname))

