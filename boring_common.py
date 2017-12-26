#!/usr/bin/env python 
from __future__ import division

import keras
import cv2
import os
import sys
import subprocess as sp

labels = ['boring', 'negative']

def preprocess_image(x):
    # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def resize_image(img, min_side=600, max_side=1024):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def ext(path, period=False):
    x = os.path.splitext(path)[1]
    if not period:
        x = x.replace('.', '')
    return x

def without_ext(path): 
    return os.path.splitext(path)[0]

# http://stackoverflow.com/a/14102014/627517
def overlay_image(l_img, s_img, x_offset=0, y_offset=0):
    assert y_offset + s_img.shape[0] <= l_img.shape[0]
    assert x_offset + s_img.shape[1] <= l_img.shape[1]

    l_img = l_img.copy()
    for c in range(0, 3):
        l_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1], c] = (
                  s_img[:,:,c] * (s_img[:,:,3]/255.0) +
                  l_img[y_offset:y_offset+s_img.shape[0],
                        x_offset:x_offset+s_img.shape[1], c] *
                  (1.0 - s_img[:,:,3]/255.0))
    return l_img

def mkv_to_mp4(mkv_path, remove_mkv=False, has_audio=True, quiet=True):
    assert os.path.isfile(mkv_path)
    assert ext(mkv_path) == 'mkv'
    mp4_path = without_ext(mkv_path) + '.mp4'
    
    if os.path.isfile(mp4_path):
        os.remove(mp4_path)

    #audio_codec_string = '-c:a libfdk_aac -b:a 128k' if has_audio else '-c:an'
    audio_codec_string = '-acodec copy' if has_audio else '-an'

    quiet_str = '>/dev/null 2>&1' if quiet else ''
    cmd = 'ffmpeg -i {} -vcodec copy {} {} {}'.format(
        mkv_path, audio_codec_string, mp4_path, quiet_str)
    if not quiet:
        print cmd
    sp.call(cmd, shell=True)

    assert os.path.isfile(mp4_path) # make sure that the file got generated successfully

    if remove_mkv:
        assert os.path.isfile(mkv_path)
        os.remove(mkv_path)
