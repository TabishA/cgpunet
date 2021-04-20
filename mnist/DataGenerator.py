#DataGenerator.py
from dataHelperTif import *
import numpy as np
from PIL import Image
import tensorflow as tf
keras = tf.compat.v1.keras
#import keras.utils.Sequence

class DataGenerator(keras.utils.Sequence):

    def __init__(self, slices_fn, segments_fn, batch_size=1, input_shape=(128,128,1), target_shape=(128,128,1), shuffle=True):
        self.slices_fn = slices_fn
        self.segments_fn = segments_fn
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.on_epoch_end()

    
    def __len__(self):
        return int(np.floor(len(self.slices_fn)/self.batch_size))

    
    def __getitem__(self, index):
        # Generate batch indices
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        slices_fn_temp = [self.slices_fn[k] for k in indexes]

        X,Y = self.__data_generation(slices_fn_temp)

        return X,Y
    
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.slices_fn))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    
    def __data_generation(self, slices_fn_temp):
        
        X = np.empty((self.batch_size, *self.input_shape))
        Y = np.empty((self.batch_size, *self.target_shape))

        for i, ID in enumerate(slices_fn_temp):
            img = np.array(Image.open(ID))
            img = np.reshape(img, self.input_shape)/255
            seg = np.array(Image.open(self.segments_fn[ID]))
            seg = np.reshape(seg, self.target_shape)/255
            seg[seg>0.5]=1
            seg[seg<=0.5]=0

            X[i,] = img
            Y[i,] = seg

        return X,Y 



