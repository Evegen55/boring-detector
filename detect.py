#!/usr/bin/env python 
from __future__ import division

import keras
import keras.preprocessing.image
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.preprocessing.csv_generator import CSVGenerator

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import sys
from collections import OrderedDict
import subprocess as sp

import tensorflow as tf
import boring_common as bcm

# paths
boring_repository_dir = os.getcwd() # you can set this manually if you're not calling the script from the top repo directory
boring_snapshots_dir = os.path.join(boring_repository_dir, 'boring-snapshots')
boring_model_path = os.path.join(boring_snapshots_dir, 'boring_resnet50_best.h5')
out_dir = os.path.join(boring_repository_dir, 'output/detection')
in_dir = os.path.join(boring_repository_dir, 'boring-dataset/boring')
design_dir = os.path.join(boring_repository_dir, 'showcase/design')

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

keras.backend.tensorflow_backend.set_session(get_session())
model = keras.models.load_model(boring_model_path, custom_objects=custom_objects)

# load in image resources
img_boring = cv2.imread(os.path.join(design_dir, 'decor_boring.png'), cv2.IMREAD_UNCHANGED)
img_negative = cv2.imread(os.path.join(design_dir, 'decor_negative.png'), cv2.IMREAD_UNCHANGED)
img_lex = cv2.imread(os.path.join(design_dir, 'decor_lex.png'), cv2.IMREAD_UNCHANGED)
    
# Take in an RGB image, preprocess, and perform the inference. Returned are the max score (in case no detections) and
# the list of all detections. This is the core detection function that other wrappers call.
def detect_in_opencv_image(img):
    assert img.shape[2] == 3

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # preprocess image for network
    img = bcm.preprocess_image(img)
    img, scale = bcm.resize_image(img)

    # process image
    start_secs = time.time()
    _, _, detections = model.predict_on_batch(np.expand_dims(img, axis=0))
    dur_secs = time.time() - start_secs

    # compute predicted labels and scores
    predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
    scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]

    # correct for image scale
    detections[0, :, :4] /= scale

    # visualize detections
    boring_count = 0
    max_score = -1

    out = OrderedDict()
    dets = []
    
    for idx, (label_id, score) in enumerate(zip(predicted_labels, scores)):
        max_score = max(max_score, score)
        if score < 0.5:
            continue

        b = detections[0, idx, :4].astype(int)
        assert len(b) == 4

        det = OrderedDict()
        det['bounding_box'] = b
        det['score'] = score
        det['label'] = bcm.labels[label_id]
        dets.append(det)

    return max_score, dets

def draw_detections(img, max_score, dets):
    img = img.copy()   

    for det in dets:
        x1, y1, x2, y2 = det['bounding_box']
        score = det['score']
        label = det['label']

        # for drawing purposes it looks weird if a bounding box goes outside the image, so let's fix it
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1]-1), min(y2, img.shape[0]-1)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)

    img = bcm.overlay_image(img, img_lex)
    if len(dets) > 0:
        img = bcm.overlay_image(img, img_boring)
        font_color = (255, 255, 255)
    else:
        img = bcm.overlay_image(img, img_negative)
        font_color = (0, 0, 255)

    text = "Boring Score:".format(round(max_score, 4))
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color , 2, cv2.LINE_AA)

    text = "{}".format(round(max_score, 4))
    cv2.putText(img, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, font_color , 3, cv2.LINE_AA)

        
    return img
    
def detect_in_video_file(in_vid_path, out_dir, generate_gif=False):
    assert os.path.isdir(out_dir)
    assert os.path.isfile(in_vid_path), in_vid_path
    assert bcm.ext(in_vid_path) != '', 'input file path should be a video file: {}'.format(in_vid_path)
    vid_name = bcm.without_ext(os.path.basename(in_vid_path))
    out_mkv_path = os.path.join(out_dir, '{}-detected.mkv'.format(vid_name))

    cap = cv2.VideoCapture(in_vid_path)
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    vid_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) # cap property returns float so we round and convert to int
    vid_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_width_height = (vid_width, vid_height)
    assert vid_width > 0 and vid_height > 0
    assert fps > 0

    vw = cv2.VideoWriter(out_mkv_path, fourcc, fps, vid_width_height)

    fi = -1
    while(cap.isOpened()):
        fi += 1
        ret, img = cap.read()
        if not ret:
            break

        # if fi >= 30:
        #     break
        
        max_score, dets = detect_in_opencv_image(img)
        img = draw_detections(img, max_score, dets)
        
        vw.write(img)
        
    cap.release()
    vw.release()

    if generate_gif:
        cmd = 'gifify --output {}.gif --resize -1:360 {}'.format(bcm.without_ext(out_mkv_path), out_mkv_path)
        sp.call(cmd, shell=True)

    bcm.mkv_to_mp4(out_mkv_path, remove_mkv=True, has_audio=False, quiet=True)

if __name__ == '__main__':
    in_dir = os.path.join(boring_repository_dir, 'showcase/vids_input')
    out_dir = os.path.join(boring_repository_dir, 'showcase/vids_output')

    vid_names = [x for x in os.listdir(in_dir) if bcm.ext(x) in ['mp4', 'avi', 'mkv', 'mov']]
    #vid_names = ['boring-hat-fast.mp4']
    
    for vid_name in vid_names:
        if bcm.ext(vid_name) not in ['mp4', 'avi', 'mkv', 'mov']:
            continue
        in_vid_path = os.path.join(in_dir, vid_name)
        detect_in_video_file(in_vid_path, out_dir, generate_gif=True)
