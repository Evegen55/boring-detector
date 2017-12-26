#!/usr/bin/env python 
from __future__ import division

import argparse
import os
from pprint import pprint

import keras
import keras.preprocessing.image

import tensorflow as tf

import keras_retinanet.losses
import keras_retinanet.layers
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.models.resnet import ResNet50RetinaNet
from keras_retinanet.utils.keras_version import check_keras_version

# parameters
batch_size = 1
steps_per_epoch = 10000
epochs = 50
# steps_per_training_epoch = 1500
# training_epochs = 10

# paths
boring_repository_dir = '/home/lex/Dropbox/projects/mit/code/boring-detector'
boring_dataset_dir = os.path.join(boring_repository_dir, 'boring-dataset')
boring_annotations_path = os.path.join(boring_dataset_dir, 'boring-images.csv')
boring_classes_path = os.path.join(boring_dataset_dir, 'boring-classes.csv')
boring_snapshots_dir = os.path.join(boring_repository_dir, 'boring-snapshots')

assert os.path.isfile(boring_annotations_path), boring_annotations_path
assert os.path.isfile(boring_classes_path), boring_classes_path

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_models(num_classes):
    # create "base" model (no NMS)
    image = keras.layers.Input((None, None, 3))

    model = ResNet50RetinaNet(image, num_classes=num_classes, weights='imagenet', nms=False)
    training_model = model

    # append NMS for prediction only
    classification   = model.outputs[1]
    detections       = model.outputs[2]
    boxes            = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
    detections       = keras_retinanet.layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
    prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])

    # compile model
    training_model.compile(
        loss={
            'regression'    : keras_retinanet.losses.smooth_l1(),
            'classification': keras_retinanet.losses.focal()
        },
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model):
    callbacks = []

    # save the prediction model
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(boring_snapshots_dir, 'boring_resnet50_{epoch:02d}.h5'),
        verbose=1
    )
    checkpoint = RedirectModel(checkpoint, prediction_model)
    callbacks.append(checkpoint)

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1,
                                                     mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    callbacks.append(lr_scheduler)

    return callbacks


def create_generator():
    # create image data generator objects
    train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
    )

    train_generator = CSVGenerator(
        boring_annotations_path,
        boring_classes_path,
        train_image_data_generator,
        batch_size=batch_size
    )

    return train_generator

if __name__ == '__main__':
    # make sure keras is the minimum required version
    check_keras_version()

    keras.backend.tensorflow_backend.set_session(get_session())

    # create the generators
    train_generator = create_generator()

    # create the model
    print('Creating model, this may take a second...')
    model, training_model, prediction_model = create_models(num_classes=train_generator.num_classes())

    # print model summary
    #print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(model, training_model, prediction_model)

    # start training
    training_model.fit_generator(
        generator = train_generator,
        steps_per_epoch = steps_per_training_epoch,
        epochs = training_epochs,
        verbose=1,
        callbacks=callbacks,
    )
