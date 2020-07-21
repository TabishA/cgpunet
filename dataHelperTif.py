#dataHelperTif.py
from PIL import Image
import numpy as np
import glob
import os
import sys
import math
import shutil

def get_files(path, end):
    return glob.glob(path + end)


def rgb_2_lum(img):
    img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    return img


def normalize_img(img_rgb):
    img_norm = np.zeros(img_rgb.shape)
    for ch in range(img_rgb.shape[2]):
        img = img_rgb[:, :, ch]
        img_norm[:, :, ch] = normalize(img)
    return img_norm


def normalize(img):
    m = float(np.mean(img))
    st = float(np.std(img))
    if st > 0:
        norm = (img - m) / float(st)
    else:
        norm = img - m
    return norm


#generates tiles in order, with overlap as needed to accomodate required width
def generate_tiles_ordered(slice, segment, tile_width, offset = (128, 128)):
    slice_tiles = list()
    segment_tiles = list()

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
            slice_tiles.append(cropped_img)

            cropped_seg = segment[horiz_start:horiz_end, vert_start:vert_end]
            segment_tiles.append(cropped_seg)

            # if segment != None:
            #     cropped_seg = segment[horiz_start:horiz_end, vert_start:vert_end]
            #     segment_tiles.append(cropped_seg)

    # if segment != None:
    #     return slice_tiles, segment_tiles
    # else:
    #     return slice_tiles

    return slice_tiles, segment_tiles



def get_coord_random(low, high, num_tiles):
    return list(np.random.randint(low, high, num_tiles))



def generate_tiles_random(slice, segment, tile_width):
    num_tiles = 4*math.ceil(slice.shape[1]/tile_width)
    coord_x = get_coord_random(0, slice.shape[0] - tile_width, num_tiles)
    coord_y = get_coord_random(0, slice.shape[1] - tile_width, num_tiles)

    print('number of random tiles ' + str(num_tiles))
    print(slice.shape[0])
    print(coord_x)
    print(coord_y)

    slice_tiles_random = list()
    segment_tiles_random = list()

    for i in range(len(coord_x)):
        for j in range(len(coord_y)):
            if coord_x[i] + tile_width <= slice.shape[0]:
                horiz_start = coord_x[i]
                horiz_end = coord_x[i] + tile_width
            else:
                horiz_start = coord_x[i] - tile_width
                horiz_end = coord_x[i]

            if coord_y[j] + tile_width <= slice.shape[1]:
                vert_start = coord_y[j]
                vert_end = coord_y[j] + tile_width
            else:
                vert_start = coord_y[j] - tile_width
                vert_end = coord_y[j]

            cropped_img = slice[horiz_start:horiz_end, vert_start:vert_end]
            cropped_seg = segment[horiz_start:horiz_end, vert_start:vert_end]

            slice_tiles_random.append(cropped_img)
            segment_tiles_random.append(cropped_seg)

    return slice_tiles_random, segment_tiles_random



def save_tiles(tiles, sub_directory, out_directory, ext):
    for j in range(len(tiles)):
        img = Image.fromarray(tiles[j])
        out_name = sub_directory + str(j) + ext
        out_path = os.path.join(out_directory, out_name)
        img.save(out_path)
        print(out_path)



def generate_tiles(tile_width, path_slices, path_segments, end_slc, end_seg, out_folder, out_folder_seg):
    slices_fn = get_files(path_slices, end_slc)
    segments_fn = get_files(path_segments, end_seg)
    
    slices_fn.sort()
    segments_fn.sort()
    
    assert(len(slices_fn) == len(segments_fn))
    
    w = str(tile_width)
    
    for i in range(len(slices_fn)):
        basename = os.path.basename(slices_fn[i])
        _, ext = os.path.splitext(slices_fn[i])
        out_bn = basename.replace(ext, '.patch.' + w + '.')
        out_bn_rand = basename.replace(ext, '.patch.random.' + w + '.')
        
        basename_seg = os.path.basename(segments_fn[i])
        _, ext_seg = os.path.splitext(segments_fn[i])
        out_bn_seg = basename_seg.replace(ext_seg, '.patch.' + w + '.')
        out_bn_seg_rand = basename_seg.replace(ext_seg, '.patch.random.' + w + '.')

        slice = np.array(Image.open(slices_fn[i]))
        segment = np.array(Image.open(segments_fn[i]))
        
        
        if slice.shape[0] > tile_width and slice.shape[1] > tile_width:
            
            slice_tiles, segment_tiles = generate_tiles_ordered(slice, segment, tile_width, offset = (tile_width, tile_width))
            slice_tiles_random, segment_tiles_random = generate_tiles_random(slice, segment, tile_width)
            
        else:
            print('{0} smaller than {1}'.format(slices_fn[i], tile_width))
            continue
              
        
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
            
            
        if not os.path.isdir(out_folder_seg):
            os.mkdir(out_folder_seg)
        

        save_tiles(slice_tiles, out_bn, out_folder, ext)
        save_tiles(segment_tiles, out_bn_seg, out_folder_seg, ext_seg)
        save_tiles(slice_tiles_random, out_bn_rand, out_folder, ext)
        save_tiles(segment_tiles_random, out_bn_seg_rand, out_folder_seg, ext_seg)



def split_train_valid(path_img_patches, path_seg_patches, train_path, valid_path, end):
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    
    train_input = os.path.join(train_path, 'input')
    train_target = os.path.join(train_path, 'target')
    valid_input = os.path.join(valid_path, 'input')
    valid_target = os.path.join(valid_path, 'target')

    if not os.path.exists(train_input):
        os.makedirs(train_input)

    if not os.path.exists(valid_input):
        os.makedirs(valid_input)

    if not os.path.exists(train_target):
        os.makedirs(train_target)

    if not os.path.exists(valid_target):
        os.makedirs(valid_target)

    img_patches = get_files(path_img_patches, end)
    seg_patches = get_files(path_seg_patches, end)

    img_patches.sort()
    seg_patches.sort()

    assert(len(img_patches)==len(seg_patches))

    train_img_patches = np.random.choice(img_patches, 0.9*len(img_patches), replace=False)
    train_img_patches = list(train_img_patches)
    valid_img_patches = list()
    train_seg_patches = list()
    valid_seg_patches = list()

    for p in img_patches:
        if p not in train_img_patches:
            valid_img_patches.append(p)

    for p in train_img_patches:
        bn = os.path.basename(p)
        bn = bn.replace('_training.gray', '_manual1')
        seg = os.path.join(path_seg_patches, bn)
        train_seg_patches.append(seg)

    for p in valid_img_patches:
        bn = os.path.basename(p)
        bn = bn.replace('_training.gray', '_manual1')
        seg = os.path.join(path_seg_patches, bn)
        valid_seg_patches.append(seg)

    
    copy_list(train_img_patches, train_input)
    copy_list(train_seg_patches, train_target)
    copy_list(valid_img_patches, valid_input)
    copy_list(valid_seg_patches, valid_target)

    
def copy_list(source_list, dest_path):
    for elem in source_list:
        bn = os.path.basename(elem)
        out_path = os.path.join(dest_path, bn)
        shutil.copyfile(elem, out_path)

