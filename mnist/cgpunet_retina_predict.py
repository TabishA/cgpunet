#cgpunet_retina_predict.py
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.metrics as metrics
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

class Tile:
    def __init__(self, info, coords, tile_width):
        self.info = info
        self.coords = coords
        self.tile_width = tile_width



def generate_tiles_ordered(slice, tile_width, offset = (128, 128)):
    slice_tiles = list()
    #segment_tiles = list()

    for i in range(int(math.ceil(slice.shape[0] / (offset[1] * 1.0)))):
        for j in range(int(math.ceil(slice.shape[1] / (offset[0] * 1.0)))):

            if offset[1] * i + tile_width <= slice.shape[0]:
                horiz_start = offset[1] * i
                horiz_end = offset[1] * i + tile_width
            else:
                horiz_start = slice.shape[0] - tile_width
                horiz_end = slice.shape[0]

            if offset[0] * j + tile_width <= slice.shape[1]:
                vert_start = offset[0] * j
                vert_end = offset[0] * j + tile_width
            else:
                vert_start = slice.shape[1] - tile_width
                vert_end = slice.shape[1]

            cropped_img = slice[horiz_start:horiz_end, vert_start:vert_end]
            #cropped_seg = segment[horiz_start:horiz_end, vert_start:vert_end]
            print(cropped_img.shape)
            tile_img = Tile(cropped_img, (horiz_start, vert_start), tile_width)
            #tile_seg = Tile(cropped_seg, (horiz_start, vert_start), tile_width)

            slice_tiles.append(tile_img)
            #segment_tiles.append(tile_seg)

    return slice_tiles


def merge_tiles(tiles, slice_shape):
    merged = np.zeros(slice_shape)
    for t in tiles:
        coord_x = t.coords[0]
        coord_y = t.coords[1]
        tile_width = t.tile_width
        info = t.info
        merged[coord_x:coord_x+tile_width, coord_y:coord_y+tile_width] = info

    return merged


def get_prediction(slice_tiles, input_size=(128,128,1)):
    #model = unet(pretrained_weights='unet_retina.hdf5', input_size=input_size)
    #model = BCDU_net_D3(pretrained_weights='bcdunet_isbi.hdf5', input_size=input_size)
    model = load_model('./cgpunet_drive_6_15.hdf5')
    prediction_list = list()

    for tile in slice_tiles:
        im_arr = np.reshape(tile.info, (1,)+input_size)

        tile_pred = Tile(info=None, coords = tile.coords, tile_width = tile.tile_width)
        data_pred = np.around(model.predict(im_arr, batch_size=1))
        
        try:
            print(data_pred.shape)
        except Exception as e:
            print(str(e))
            pass

        tile_pred.info = data_pred[0,:]

        prediction_list.append(tile_pred)

    return prediction_list

if __name__ == "__main__":
    # img_path = '03_test.gray.tif'
    # img_path = os.path.dirname(img_path)
    # basename = os.path.basename(img_path)

    # img_path_out = basename.replace('gray', 'prediction')

    img = Image.open('03_test.gray.tif')
    im_arr = np.array(img)

    plt.imshow(im_arr)
    plt.show()

    im_arr = np.reshape(im_arr, (im_arr.shape[0], im_arr.shape[1], 1))/255
    slice_tiles = generate_tiles_ordered(slice = im_arr, tile_width = 128, offset=(128,128))
    predicted_tiles = get_prediction(slice_tiles, input_size=(128,128,1))

    merged = merge_tiles(predicted_tiles, im_arr.shape)
    print(merged.shape)

    pred_arr = merged[:,:,0]
    pred = Image.fromarray(pred_arr*255).convert('L')

    plt.imshow(pred)
    plt.show()
    pred.save('03.retina.prediction_cgpunet.png')
