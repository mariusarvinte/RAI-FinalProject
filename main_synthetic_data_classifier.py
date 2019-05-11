#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:09:28 2019

This script generates synthetic data and trains ReLU classifiers with increasing
number of distance embeddings.

Authors: Marius Arvinte, Mai Lee Chang and (Ethan) Yuqiang Heng
"""

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from scipy.spatial.distance import cdist
import os

# Clear Keras session
K.clear_session()

# Import from other files
from aux_layers import CosineEmbedding, EuclideanEmbedding, L1Embedding
from aux_layers import r1_crossentropy
from aux_geometry import gen_centroids, gen_synthetic_data

# Data properties
num_classes = 10
latent_dim  = 10
# Synthetic data properties
num_points = 10000 # Per class
mu = 0
sigma = 0.01

# Generate synthetic data centroids
# Generate points spread apart as much as possible
centroids = gen_centroids(latent_dim, num_classes,
                          seed=123, num_tries=int(1e5))
# Generate noise around points and label them
x_train, x_val, y_train, y_val = gen_synthetic_data(latent_dim, num_classes,
                                                    centroids, num_points,
                                                    mu, sigma)

# Embedding type
embedding_type = 'cosine'

# Determine centroid add order
# Compute all pairwise embeddings between centroids
# Special handle for L1
if embedding_type == 'l1':
    cross_dist = cdist(centroids, centroids, metric='minkowski', p=1)
else:
    cross_dist = cdist(centroids, centroids, metric=embedding_type)
# Remove diagonal terms
cross_dist = cross_dist[~np.eye(cross_dist.shape[0],
                                    dtype=bool)].reshape(cross_dist.shape[0],-1)
# Compute standard deviation of distance from each point to all others
cross_std = np.std(cross_dist, axis=-1)
# Pick max-first
p_order = np.argsort(-cross_std)

# Number of tries
num_tries = 1000

# Algorithm to use when adding next prototype
# Currently either 'proposed' or 'random'
increment_type = 'proposed'

# Deep ReLU net parameters
num_layers = 3 # Hidden
hidden_dim = num_classes # Generally unchanged
# L2 regularizer on weights
weight_reg = l2(l=1e-3)

# Training parameters
batch_size = 200
num_epochs = 5

# Seeding parameters
global_seed = 7777
# Draw local seeds
np.random.seed(global_seed)
local_seed  = np.random.randint(low=0, high=2**31, size=(num_tries))
# Global directory name
global_dir = 'deep_clf_' + increment_type + '_global' + str(global_seed) + '_hidden' + str(num_layers)
if not os.path.exists(global_dir):
    os.makedirs(global_dir)

# For each try
for try_idx in range(num_tries):
    # Clear Keras graph
    K.clear_session()

    # Local directory name
    local_dir = global_dir + '/local' + str(local_seed[try_idx]) + '_try' + str(try_idx)
    if not os.path.exists(local_dir): 
        os.makedirs(local_dir)
    # For each number of prototype vectors (up to number of classes)
    for p_idx in range(num_classes):
        # If the first one, pick at random
        if p_idx == 0:
            # Fix seed
            np.random.seed(local_seed[try_idx])
            p_label      = p_order[p_idx] # This corresponds to max spread
            p_collection = np.expand_dims(centroids[p_label], axis=0)
        # Proposed ordering in the paper
        elif increment_type=='proposed':
            new_label    = p_order[p_idx]
            p_label      = np.append(p_label, new_label)
            p_collection = np.append(p_collection, np.expand_dims(centroids[new_label], axis=0), axis=0)
        
        # Random order
        elif increment_type=='random':
            # Fix seed
            np.random.seed(local_seed[try_idx]+p_idx)
            new_label    = np.random.choice(np.setdiff1d([i for i in range(num_classes)], p_label))
            p_label      = np.append(p_label, new_label)
            p_collection = np.append(p_collection, np.expand_dims(centroids[new_label], axis=0), axis=0)
        
        # Clear graph
        K.clear_session()
        
        # Instantiate and train a deep ReLU classifier
        latent_input = Input(shape=(latent_dim, ))
        
        # Distance embedding and commitment losses
        if embedding_type == 'cosine':
            distances, r1, r2 = CosineEmbedding(num_vectors=p_idx+1,
                                                latent_dim=latent_dim)(latent_input)
        elif embedding_type == 'euclidean':
            distances, r1, r2 = EuclideanEmbedding(num_vectors=p_idx+1,
                                                   latent_dim=latent_dim)(latent_input)
        elif embedding_type == 'l1':
            distances, r1, r2 = L1Embedding(num_vectors=p_idx+1,
                                            latent_dim=latent_dim)(latent_input)
            
        # Hidden layers
        hidden = Dense(hidden_dim, activation='relu',
                       kernel_initializer=glorot_uniform(local_seed[try_idx]),
                       kernel_regularizer=weight_reg)(distances)
        for layer_idx in range(num_layers-1):
            hidden = Dense(hidden_dim, activation='relu',
                           kernel_initializer=glorot_uniform(local_seed[try_idx]+layer_idx+1),
                           kernel_regularizer=weight_reg)(hidden)
        # Final linear layer + softmax
        output = Dense(num_classes, activation='softmax',
                       kernel_initializer=glorot_uniform(local_seed[try_idx]+num_layers),
                       kernel_regularizer=weight_reg)(hidden)        

        # Create model and display summary
        deep_clf = Model(latent_input, output)
        deep_clf.summary()
        
        # Optimizer and compile
        # Use custom loss function involving R1 commitment term
        adadelta = Adadelta(lr=1.)
        deep_clf.compile(adadelta, loss=r1_crossentropy(r1, lambda_1=0.05),
                         metrics=[categorical_accuracy])
                    # Load previous weights

        # Save best weights callback
        local_filename = local_dir + '/weights_prototypes' + str(p_idx+1) + '.h5'
        bestModel = ModelCheckpoint(local_filename, monitor='val_categorical_accuracy',
                                    save_best_only=True, save_weights_only=True)
        # Slow learning rate on plateau
        slowRate = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                                     patience=10, min_lr=0.1, verbose=1)
        
        # Fit on data using limited number of prototypes
        # Fresh re-initialization
        if p_idx == 0:
            # Begin from best centroid
            deep_clf.layers[1].set_weights([p_collection])
            
            # Train
            history = deep_clf.fit(x=x_train, y=to_categorical(y_train), batch_size=batch_size,
                                   epochs=num_epochs, verbose=2, validation_split=0.2,
                                   callbacks=[bestModel, slowRate])
        else:
            # Append one new prototype given by the ordering
            new_p = p_collection[-1]
            pre_kernels = np.append(frozen_p[0], np.expand_dims(new_p, axis=0), axis=0)
            
            # Load previous weights
            deep_clf.layers[1].set_weights([pre_kernels])
            
            # Train
            history = deep_clf.fit(x=x_train, y=to_categorical(y_train), batch_size=batch_size,
                                   epochs=num_epochs, verbose=2, validation_split=0.2,
                                   callbacks=[bestModel, slowRate])
        
        # Reload best model
        deep_clf.load_weights(local_filename)
        # Save p vectors for future reload
        frozen_p = deep_clf.layers[1].get_weights()
        
        # Record validation loss in global file
        with open(global_dir + '/global_results.txt', 'a+') as file:
            file.write('Local seed %d, Try number %d, Number of prototype vectors = %d, Best val. acc = %f\n'
                       % (local_seed[try_idx], try_idx, p_idx+1, np.max(history.history['val_categorical_accuracy'])))