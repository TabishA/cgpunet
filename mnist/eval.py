#eval.py

#from cgp import *
#from cgp_2_dag import *
#from dag_2_cnn import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
from DataGenerator import *
import os
import sys
import glob
import argparse
import matplotlib as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


def get_files(path, end):
    return glob.glob(path + end)

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=dir_path)

args = parser.parse_args()

dir_name = os.path.dirname(args.path)
test_path = os.path.join(dir_name, 'data/')

test_path_img = os.path.join(test_path, 'test_input/')
test_path_mask = os.path.join(test_path, 'test_target/')

#update extensions
test_slices = get_files(test_path_img, "*.tif")
test_masks = get_files(test_path_mask, "*.tif")

if len(test_slices)==0:
    sys.exit("empty test list")


test_slices.sort()
test_masks.sort()

labels_test = dict()

for i in range(len(test_slices)):
    labels_test[test_slices[i]] = test_masks[i]


testGenerator = DataGenerator(slices_fn=test_slices, segments_fn=labels_test,batch_size=16, input_shape=(128,128,1), target_shape=(128,128,1))

#G = nx.read_gpickle('./p_files/cgpunet_drive_6_15.gpickle')
#model = dag_2_cnn(G, 0, input_shape=(128,128,1), target_shape=(128,128,1), pretrained_weights='./saved_models/cgpunet_drive_6_15.hdf5') 

model = load_model('./cgpunet_drive_6_15_full.hdf5')
model.summary()

eval_metrics = model.evaluate_generator(testGenerator, steps=int(len(test_slices)/16), use_multiprocessing=True)
print(model.metrics_names)
print (eval_metrics)

