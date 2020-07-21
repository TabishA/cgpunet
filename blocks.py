#blocks.py
from keras.layers import *
import math


# addition: if channel numbers can be broadcasted, add layers and return.
# else increase the number of channels in the smaller feature map by 1x1 convolution -> batchnorm (add_mode='conv')
# TODO: add_mode='zeros' - zero padding instead of convolving to reduce number of trainable parameters
# returns: sum tensor, convolution flag (0 if no conv, else 1)
def add_layers(A, B, add_mode='conv', add_name='add'):
    a_channels = tuple(A.shape)[3]
    b_channels = tuple(B.shape)[3]
    if a_channels == b_channels or a_channels == 1 or b_channels == 1:
        return add([A,B], name=add_name), 0
    elif a_channels > b_channels:
        B = Conv2D(a_channels, kernel_size=(1,1), padding="same", name=add_name + '_Conv1x1', kernel_initializer='he_normal', use_bias=False)(B)
        B = BatchNormalization(axis=3, name=add_name + '_Conv1x1_bn' )(B)
        return add([A,B], name=add_name), 1
    else:
        A = Conv2D(b_channels, kernel_size=(1,1), padding="same", name=add_name + '_Conv1x1', kernel_initializer='he_normal', use_bias=False)(A)
        A = BatchNormalization(axis=3, name=add_name + '_Conv1x1_bn')(A)
        return add([A,B], name=add_name), 1


#concatenates two layers if equal shape, otherwise larger layer is downsampled by max_pooling
#this version assumes size of layers is related by a factor of two
def MergeBlock(layer_a, layer_b, mode='concat', block_name='merge'):
    
    def layer(A, B, max_pool=0, to_downsample=None):
        #print('pool factor: {}, to downsample: {}'.format(max_pool, to_downsample))
        
        conv_flag = 0
        
        if max_pool == 0:
            if mode == 'add':
                x, conv_flag = add_layers(A, B, add_name=block_name + '_add')
            else:
                x = concatenate([A,B], axis=3, name=block_name + '_concat')
            if conv_flag: x = Activation('relu', name=block_name + '_add_relu')(x)
            return x
        elif to_downsample == 0:
            for _ in range(max_pool):
                A = MaxPooling2D(pool_size=(2,2), name=block_name + '_max_pool')(A)
                #print('Max Pooling - {}.shape: {}'.format(A, A.shape))
        else:
            for _ in range(max_pool):
                B = MaxPooling2D(pool_size=(2,2), name=block_name + '_max_pool')(B)
                #print('Max Pooling - {}.shape: {}'.format(B, B.shape))

        if mode == 'add':
            x, conv_flag = add_layers(A, B, add_name=block_name + '_add')
        else:
            x = concatenate([A,B], axis=3, name=block_name + '_concat')
        
        if conv_flag: x = Activation('relu', name=block_name + '_add_relu')(x)
        
        return x

    shape_a = tuple(layer_a.shape)
    shape_b = tuple(layer_b.shape)
    if ((shape_a[1], shape_a[2]) == (shape_b[1], shape_b[2])):
        return layer(layer_a, layer_b)
    elif (shape_a[1]/shape_b[1] > 1):
        mult_factor = int(shape_a[1]/shape_b[1])
        pool_factor = int(math.log2(mult_factor))
        return layer(layer_a, layer_b, max_pool=pool_factor, to_downsample=0)
    else:
        mult_factor = int(shape_b[1]/shape_a[1])
        pool_factor = int(math.log2(mult_factor))
        return layer(layer_a, layer_b, max_pool=pool_factor, to_downsample=1)



#https://github.com/MrGiovanni/UNetPlusPlus/blob/master/segmentation_models/xnet/blocks.py
def ConvBlock(filters, kernel_size, use_batchnorm=True, conv_name='conv'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name + '_Conv2D', kernel_initializer='he_normal', use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(axis=3, name=conv_name + '_bn')(x)
        if kernel_size == (1, 1):
            x = Activation('sigmoid', name=conv_name + '_sigmoid')(x)
        else:
            x = Activation('relu', name=conv_name + '_relu')(x)
        return x
    return layer



def ResBlock(input_layer, filters, kernel_size, use_batchnorm=True, res_name='res'):
    def layer(x):
        x = ConvBlock(filters, kernel_size, use_batchnorm, conv_name=res_name + '_ConvBlock')(input_layer)
        x = Conv2D(filters, kernel_size, padding="same", name=res_name + '_Conv2D', kernel_initializer='he_normal', use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = BatchNormalization(axis=3, name=res_name + '_bn')(x)
        x, _ = add_layers(x, input_layer, add_name=res_name + '_add')
        x = Activation('relu', name=res_name + '_relu')(x)
        return x
    
    return layer


def DeconvBlock(input_layer, filters, kernel_size, use_batchnorm=True, deconv_name='deconv'):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=deconv_name + '_Conv2D', kernel_initializer='he_normal', use_bias=not(use_batchnorm))(input_layer)
        if use_batchnorm:
            x = BatchNormalization(axis=3, name=deconv_name + '_bn')(x)
        x = Activation('relu', name=deconv_name + '_relu')(x)
        x = UpSampling2D(size=(2,2))(x)

        return x

    return layer
        


#adds two layers if equal shape, otherwise larger layer is downsampled by max_pooling
#this version assumes size of layers is related by a factor of two
# def AddBlock(layer_a, layer_b):

#     def layer(A, B, max_pool=0, to_downsample=None):
#         print('pool factor: {}, to downsample: {}'.format(max_pool, to_downsample))
#         if max_pool == 0:
#             x = add([A,B])
#             return x
#         elif to_downsample == 0:
#             for _ in range(max_pool):
#                 A = MaxPooling2D(pool_size=(2,2))(A)
#                 print('Max Pooling - {}.shape: {}'.format(A, A.shape))
#         else:
#             for _ in range(max_pool):
#                 B = MaxPooling2D(pool_size=(2,2))(B)
#                 print('Max Pooling - {}.shape: {}'.format(B, B.shape))

#         x = add([A,B])
#         return x

#     shape_a = tuple(layer_a.shape)
#     shape_b = tuple(layer_b.shape)
#     if ((shape_a[1], shape_a[2]) == (shape_b[1], shape_b[2])):
#         return layer(layer_a, layer_b)
#     elif (shape_a[1]/shape_b[1] > 1):
#         mult_factor = int(shape_a[1]/shape_b[1])
#         pool_factor = int(math.log2(mult_factor))
#         return layer(layer_a, layer_b, max_pool=pool_factor, to_downsample=0)
#     else:
#         mult_factor = int(shape_b[1]/shape_a[1])
#         pool_factor = int(math.log2(mult_factor))
#         return layer(layer_a, layer_b, max_pool=pool_factor, to_downsample=1)
