#cgpunet_train.py

import os
import sys
from dag_2_cnn import *
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, History
import matplotlib.pyplot as plt

# __call__: training the CNN defined by DAG
class CNN_train():

    def __init__(self, x_train, y_train, x_valid, y_valid, input_shape, verbose=True, batchsize=32, batchsize_valid=32):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.verbose = verbose
        self.batchsize = batchsize
        self.batchsize_valid = batchsize_valid
        self.input_shape = input_shape
    
    def __call__(self, dag, gpuID, epoch_num=100, out_model='cgpunet.hdf5'):
        
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', epoch_num)
            print('batch_size:', self.batchsize)
        
        model = dag_2_cnn(dag, gpuID, self.input_shape)

        #print summary
        model.summary()

        model_checkpoint = ModelCheckpoint(out_model, monitor='loss',verbose=1, save_best_only=True)
        history = History()
        
        history = model.fit(x=self.x_train, y=self.y_train, epochs=epoch_num, callbacks=[model_checkpoint], validation_data=(self.x_valid, self.y_valid), validation_freq= int(epoch_num))
        
        val_acc = history.history['val_accuracy']
        #val_loss = history.history['val_loss']
        val_precision = history.history['val_precision']
        val_recall = history.history['val_recall']

        val_f1 = 2*((val_precision[0]*val_recall[0])/(val_precision[0]+val_recall[0]+K.epsilon()))
        trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))

        if not os.path.isdir('./figures'):
            os.makedirs('./figures')

        acc_fig_name = out_model.replace('.hdf5', '_acc.png')
        acc_fig_name = './figures/' + acc_fig_name

        loss_fig_name = out_model.replace('.hdf5', '_loss.png')
        loss_fig_name = './figures/' + loss_fig_name

        # Plot training & validation accuracy values
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy: {}'.format(out_model))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(acc_fig_name)
        
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss": {}'.format(out_model))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(loss_fig_name)

        pickle_name = out_model.replace('.hdf5', '.gpickle')
        
        if not os.path.isdir('./p_files'):
            os.makedirs('./p_files')

        pickle_name = './p_files/' + pickle_name
        nx.write_gpickle(dag, pickle_name)

        K.clear_session()

        return (float(val_f1), trainable_count)
