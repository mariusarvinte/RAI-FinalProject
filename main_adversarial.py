# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:45:33 2019

@author: ma56473
"""

from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import scipy.io as sio

import os
os.environ['PYTHONHASHSEED']=str(0)

# Configure session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Auxiliary functions from other files
from aux_geometry import gen_centroids
from aux_networks import SupervisedAAE
from aux_preprocessing import load_coil20

# Clear session
K.clear_session()
plt.close('all')

## Overarching dataset
# Can be either 'mnist' or 'coil-20'
data_type = 'coil20'

# Different behaviours and some parameters
# Train, validation and test data must come out from this loop
if data_type == 'mnist':
    # Load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Expand to 4-th channel (grayscale images only)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test, axis=-1)
    
    # Normalize
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    
    # Further draw validation from training
    # 90/10 split
    x_train, x_val = np.split(x_train, [int(0.8*np.shape(x_train)[0])])
    y_train, y_val = np.split(y_train, [int(0.8*np.shape(y_train)[0])])
    
    # Dataset parameters
    num_classes = 10
    img_shape   = (28, 28, 1)
    
    # Architecture parameters
    arch = 'FC' # This is a major parameter
    # If going with arch='FC', then hidden_dim needs to be >> nr. of pixels in image
    latent_dim = 10
    hidden_dim = 1200
    num_layers  = 4 # Per encoder, decoder and discriminator each
    filter_size = 3
    
    # Training parameters
    batch_size = 200
    num_epochs = 200
    
elif data_type == 'coil20':
    x, y = load_coil20('coil-20-proc')
    x = np.asarray(x)
    x = np.expand_dims(x, axis=-1)
    y = np.asarray(y) - 1
    y = np.expand_dims(y, axis=-1)
    
    # Train/validation split
    x_train, x_val = np.split(x, [int(0.8*np.shape(x)[0])])
    y_train, y_val = np.split(y, [int(0.8*np.shape(y)[0])])
    
    y_train = np.squeeze(y_train)
    y_val = np.squeeze(y_val)

    # Dataset parameters
    num_classes = 20
    img_shape   = (128, 128, 1)
    
    # Architecture parameters
    arch = 'Conv' # This is a major parameter
    latent_dim = 10
    # If going with arch='FC', then hidden_dim needs to be >> nr. of pixels in image
    hidden_dim = 64
    num_layers  = 5 # Per encoder, decoder and discriminator each
    filter_size = 3
    
    # Training parameters
    batch_size = 32
    num_epochs = 200

## Operating mode
# This is the most important setting
# If set to 'blank', it will train a number of autoencoders (no prototypes)
# If set to 'pretrained', it will preload an autoencoder from a specific
# seed set and start adding prototype vectors
#op_mode = 'pretrained'
op_mode = 'blank'
# How many autoencoders will be trained in 'blank' mode
num_tries = int(100)

## Seeding
# This is the most important seed, everything is drawn from it
global_seed = 5678 # Make sure to change this if you run on multiple PCs
# Seed for centroid computation
centroid_seed = 123 # Don't change!
# Seed for batching
batch_seed    = 321 # Don't change!
# Seed for weight initialization
np.random.seed(global_seed)
weight_seed = np.random.randint(low=0, high=2**31, size=num_tries)


## Save-to-file frequency in epochs
# Note that this also performs TSNE at each save which is CPU heavy
save_interval = 10

## Target centroids
# We want them as spread apart from each other
# Optimize the max-min pairwise distance via random draws
# This takes a while, but only needs to be done once
centroids = gen_centroids(latent_dim, num_classes,
                          seed=centroid_seed, num_tries=int(1e5))

# In this mode, search for the best adversarial autoencoder
if op_mode == 'blank':
    # Try multiple weight restarts
    for try_idx in range(num_tries):
        # Seed tensorflow and numpy
        tf.set_random_seed(weight_seed[try_idx])
        tf.random.set_random_seed(weight_seed[try_idx])
        np.random.seed(weight_seed[try_idx])
        random.seed(weight_seed[try_idx])
        
        # Clear graph and figures
        K.clear_session()
        tf.reset_default_graph()
        plt.close('all')
        
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)

        # Seed tensorflow and numpy
        tf.set_random_seed(weight_seed[try_idx])
        tf.random.set_random_seed(weight_seed[try_idx])
        np.random.seed(weight_seed[try_idx])
        
        # Shuffle centroids with fixed permutations
        np.random.seed(try_idx)
        np.random.shuffle(centroids)
        
        # Instantiate
        ann = SupervisedAAE(latent_dim, num_classes, hidden_dim, num_layers,
                            filter_size, centroids, img_shape, global_seed,
                            weight_seed[try_idx], centroid_seed, batch_seed,
                            try_idx, arch)

        # Seed tensorflow and numpy
        tf.set_random_seed(weight_seed[try_idx])
        tf.random.set_random_seed(weight_seed[try_idx])
        np.random.seed(weight_seed[try_idx])

        # Train
        ann.train(x_train, y_train, x_val, y_val,
                  batch_size, num_epochs, save_interval)
        
# In this mode, load a specific pretrained autoencoder        
elif op_mode == 'pretrained':
    # Instantiate
    ann = SupervisedAAE(latent_dim, num_classes, hidden_dim, num_layers,
                        filter_size, centroids, img_shape, global_seed,
                        weight_seed[0], centroid_seed, batch_seed,
                        0, arch)
    
    # Load weights given by global/weight seed pair
    target_global = 5678
    target_weight = 26474647
    target_try    = 1
    target_epoch  = 60 # Load at specific epoch, where clustering is best
    
    # Load weights only for autoencoder part, adversarial part is not needed
    ann.autoencoder.load_weights('restart_results/global' + str(target_global) +
                                 '/weight' + str(target_weight) + '_try' + str(target_try) + '/ae_weights_epoch' +
                                 str(target_epoch) + '.h5')
    
    # Predict training and validation data
    latent_train = ann.encoder.predict(x_train)
    latent_val   = ann.encoder.predict(x_val)
    
    # Save to .mat
    sio.savemat(data_type + '_latent_data.mat', {'x_train':x_train, 'x_val':x_val,
                                                 'latent_train':latent_train, 'latent_val':latent_val,
                                                 'y_train':y_train, 'y_val':y_val})