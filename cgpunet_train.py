#cgpunet_train.py

import os
import sys
from dag_2_cnn import *
from tensorflow.compat.v1.keras import backend as K
#from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, History
#from keras.callbacks import ModelCheckpoint, History
from DataGenerator import *
import matplotlib.pyplot as plt

# __init__: load dataset
# __call__: training the CNN defined by DAG
class CNN_train():

    def __init__(self, dataset_path, img_format, mask_format, verbose=True, input_shape=(256,256,1), target_shape=(256,256,1), batchsize=16, batchsize_valid=16):
        self.dataset_path = dataset_path
        self.img_format = img_format
        self.mask_format = mask_format
        self.verbose = verbose
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.batchsize = batchsize
        self.batchsize_valid = batchsize_valid

        #get directory name
        #dir_name = os.path.dirname(self.dataset_path)
        dir_name = self.dataset_path
        
        train_path = os.path.join(dir_name, 'data/train')
        train_path_input = os.path.join(train_path, 'input/')
        train_path_target = os.path.join(train_path, 'target/')
        train_imgs = get_files(train_path_input, img_format)
        train_masks = get_files(train_path_target, mask_format)
        train_imgs.sort()
        train_masks.sort()

        if len(train_imgs) == 0:
            sys.exit('empty train list: {}'.format(train_path))

        
        self.train_len = len(train_imgs)

        labels_train = dict()

        for i in range(len(train_imgs)):
            labels_train[train_imgs[i]] = train_masks[i]

            
        valid_path = os.path.join(dir_name, 'data/valid')
        valid_path_input = os.path.join(valid_path, 'input/')
        valid_path_target = os.path.join(valid_path, 'target/')
        valid_imgs = get_files(valid_path_input, img_format)
        valid_masks = get_files(valid_path_target, mask_format)
        valid_imgs.sort()
        valid_masks.sort()

        labels_valid = dict()

        for i in range(len(valid_imgs)):
            labels_valid[valid_imgs[i]] = valid_masks[i]
            
        self.valid_len = len(valid_imgs)

        self.trainGenerator = DataGenerator(slices_fn=train_imgs, segments_fn=labels_train,batch_size=self.batchsize, input_shape=self.input_shape, target_shape=self.target_shape)
        self.validGenerator = DataGenerator(slices_fn=valid_imgs, segments_fn=labels_valid, batch_size=self.batchsize_valid, input_shape=self.input_shape, target_shape=self.target_shape)

    
    def __call__(self, dag, gpuID, epoch_num=100, out_model='cgpunet.hdf5'):
        
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', epoch_num)
            print('batch_size:', self.batchsize)

        
        train_steps = int(self.train_len/self.batchsize)
        valid_steps = int(self.valid_len/self.batchsize_valid)
        
        model = dag_2_cnn(dag, gpuID, self.input_shape, self.target_shape)

        #print summary
        model.summary()

        model_checkpoint = ModelCheckpoint(out_model, monitor='loss',verbose=1, save_best_only=True)
        history = History()
        
        #NOTE: default values: workers=1, multiprocessing=False.
        #TODO: investigate workers>1
        history = model.fit_generator(generator=self.trainGenerator, steps_per_epoch=train_steps, epochs=epoch_num, callbacks=[model_checkpoint], validation_data=self.validGenerator, validation_steps=valid_steps, validation_freq= int(epoch_num))
        
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
