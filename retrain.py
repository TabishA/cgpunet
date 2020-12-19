#eval.py

#from cgp import *
#from cgp_2_dag import *
#from dag_2_cnn import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping
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
train_path = os.path.join(dir_name, 'data/train')
train_path_input = os.path.join(train_path, 'input/')
train_path_target = os.path.join(train_path, 'target/')

train_imgs = get_files(train_path_input, "*.tif")
train_masks = get_files(train_path_target, "*.tif")

if len(train_imgs)==0:
    sys.exit("empty test list")


train_imgs.sort()
train_masks.sort()

labels_train = dict()

for i in range(len(train_imgs)):
    labels_train[train_imgs[i]] = train_masks[i]


valid_path = os.path.join(dir_name, 'data/valid')
valid_path_input = os.path.join(valid_path, 'input/')
valid_path_target = os.path.join(valid_path, 'target/')
valid_imgs = get_files(valid_path_input, '*.tif')
valid_masks = get_files(valid_path_target, '*.tif')
valid_imgs.sort()
valid_masks.sort()

labels_valid = dict()

for i in range(len(valid_imgs)):
    labels_valid[valid_imgs[i]] = valid_masks[i]

trainGenerator = DataGenerator(slices_fn=train_imgs, segments_fn=labels_train,batch_size=16, input_shape=(128,128,1), target_shape=(128,128,1))
validGenerator = DataGenerator(slices_fn=valid_imgs, segments_fn=labels_valid,batch_size=16, input_shape=(128,128,1), target_shape=(128,128,1))

#G = nx.read_gpickle('./p_files/cgpunet_drive_6_15.gpickle')
#model = dag_2_cnn(G, 0, input_shape=(128,128,1), target_shape=(128,128,1), pretrained_weights='./saved_models/cgpunet_drive_6_15.hdf5') 

model = load_model('./saved_models/cgpunet_drive_6_15.hdf5')
model.summary()

model_checkpoint = ModelCheckpoint('./cgpunet_drive_6_15_full.hdf5', monitor='loss',verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', restore_best_weights=True)
model.fit(trainGenerator, steps_per_epoch=int(len(train_imgs)/16), epochs=100, callbacks=[model_checkpoint, early_stopping], validation_data=validGenerator, validation_steps=int(len(valid_imgs)/16), validation_freq=1, use_multiprocessing=True, workers=22)


