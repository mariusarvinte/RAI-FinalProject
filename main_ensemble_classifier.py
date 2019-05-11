#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:25:38 2019

This script loads pre-generated clustered embeddings, trains individual
distance embedding classifiers and evaluate the performance of their ensemble

Authors: Marius Arvinte, Mai Lee Chang and (Ethan) Yuqiang Heng
"""

import numpy as np
from scipy.io import loadmat
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
import tensorflow as tf
import os
from matplotlib import pyplot as plt

# Tensorflow memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

K.clear_session()

# Import from other files
from aux_layers import CosineEmbedding, EuclideanEmbedding, L1Embedding 
from aux_layers import r1_crossentropy
from aux_geometry import find_data_centroid

# Overarching dataset
data_type = 'mnist'

# Parameters for different datasets
if data_type == 'mnist':
    # Load latent space data
    data = loadmat('mnist_latent_data.mat')
    
    # Data properties
    num_classes = 10
    latent_dim  = 10
    
    # Embedding to use
    embedding_type = 'cosine'
    
elif data_type == 'coil-20':
    # Load latent space data
    data = loadmat('coil20_latent_data.mat')
    
    # Data properties
    num_classes = 20
    latent_dim  = 10
    
    # Embedding to use
    embedding_type = 'euclidean'

# Extract variables
x_train = data['latent_train']
x_val   = data['latent_val']
y_train = data['y_train'].T
y_val   = data['y_val'].T

# Normalize and remember statistics (will be used to revert prototypes later)
x_mean = np.mean(x_train)
x_var  = np.std(x_train)
# Normalize based on training data
x_train = (x_train - x_mean) / x_var
x_val   = (x_val - x_mean) / x_var

# Compute centroids
centroids = find_data_centroid(x_train, y_train, num_classes, latent_dim)

# Determine centroid add order
# Compute all pairwise embeddings between centroids
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

# Deep ReLU net parameters
num_layers = 3 # Hidden
hidden_dim = num_classes # Generally unchanged
# L2 regularizer on weights
weight_reg = l2(l=1e-3)

# Training parameters
batch_size = 200
num_epochs = 10

# Seeding parameters
global_seed = 9991
# Draw local seeds
np.random.seed(global_seed)
local_seed  = np.random.randint(low=0, high=2**31, size=(num_tries))
# Global directory name
global_dir = 'deep_clf_mixture_global' + \
             str(global_seed) + '_hidden' + str(num_layers)
if not os.path.exists(global_dir):
    os.makedirs(global_dir)

# For each try
for try_idx in range(num_tries):
    # Clear Keras graph
    K.clear_session()

    # Predicted probabilities
    soft_output_val = np.zeros((x_val.shape[0], num_classes, num_classes))

    # Local directory name
    local_dir = global_dir + '/local' + str(local_seed[try_idx]) + '_try' + str(try_idx)
    if not os.path.exists(local_dir): 
        os.makedirs(local_dir)
    # For each number of prototype vectors (up to number of classes)
    for p_idx in range(num_classes):
        # Create classifier for k-th best p vector
        p_label      = p_order[p_idx] # This corresponds to max spread
        p_collection = np.expand_dims(centroids[p_label], axis=0)
        
        # Clear graph
        K.clear_session()
        K.tensorflow_backend.set_session(tf.Session(config=config))
        
        # Instantiate and train a deep ReLU classifier
        latent_input = Input(shape=(latent_dim, ))
        
        # Pass through prototype layer and return commitment terms
        if embedding_type == 'cosine':
            distances, r1, r2 = CosineEmbedding(num_vectors=1,
                                                latent_dim=latent_dim)(latent_input)
        elif embedding_type == 'euclidean':
            distances, r1, r2 = EuclideanEmbedding(num_vectors=1,
                                                   latent_dim=latent_dim)(latent_input)
        elif embedding_type == 'l1':
            distances, r1, r2 = L1Embedding(num_vectors=1,
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
        # Lambda = 1 effectively pins down the p vectors to data points
        adadelta = Adadelta(lr=1.)
        deep_clf.compile(adadelta, loss=r1_crossentropy(r1, lambda_1=1.),
                         metrics=[categorical_accuracy])
                    # Load previous weights

        # Save best weights callback
        local_filename = local_dir + '/weights_prototypes' + str(p_idx+1) + '.h5'
        bestModel = ModelCheckpoint(local_filename, monitor='val_categorical_accuracy',
                                    save_best_only=True, save_weights_only=True)
        # Slow learning rate on plateau
        slowRate = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                                     patience=10, min_lr=0.1, verbose=1)
        
        # Fit on data using single prototype
        # Fresh re-initialization
        deep_clf.layers[1].set_weights([p_collection])
        
        # Train
        history = deep_clf.fit(x=x_train, y=to_categorical(y_train), batch_size=batch_size,
                               epochs=num_epochs, verbose=2, validation_split=0.2,
                               callbacks=[bestModel, slowRate])
        
        # Predict validation data for future ensemble use
        soft_output_val[:, p_order[p_idx], :] = deep_clf.predict(x_val)
        
        # Record validation loss in global file
        with open(global_dir + '/global_results.txt', 'a+') as file:
            file.write('Local seed %d, Try number %d, Prototype number %d, Best val. acc = %f\n'
                       % (local_seed[try_idx], try_idx, p_idx+1, np.max(history.history['val_categorical_accuracy'])))
    
    # After all 1-D classifiers are trained, evaluate performance of ensemble
    # Outputs
    maj_dice_vote = np.zeros((len(p_order), x_val.shape[0]))
    max_vote      = np.zeros((len(p_order), x_val.shape[0]))
    average_vote  = np.zeros((len(p_order), x_val.shape[0]))
    median_vote   = np.zeros((len(p_order), x_val.shape[0]))
        
    # Hard votes
    hard_output_val = np.argmax(soft_output_val, axis=-1)
    
    # Evaluate performance when incrementally adding classifiers
    for num_experts in range(num_classes):
        # Truncate votes
        local_hard_votes = hard_output_val[:, p_order[:(num_experts+1)]]
        local_soft_votes = soft_output_val[:, p_order[:(num_experts+1)], :]    
        # Mixture of experts
        # Some methods need loop over points
        for point_idx in range(hard_output_val.shape[0]):
            # Hard voting
            local_count = np.bincount(local_hard_votes[point_idx], minlength=num_classes)
            vote_winner = np.argwhere(local_count == np.max(local_count))
            # Break ties
            if len(vote_winner) > 1:
                # With a dice roll
                maj_dice_vote[num_experts, point_idx] = int(vote_winner[np.random.randint(low=0, high=len(vote_winner))])
            else:
                maj_dice_vote[num_experts, point_idx] = int(vote_winner)
                
            # Max voting
            # Find most confident expert by comparing their most confident predictions
            most_confident = np.argmax(np.max(local_soft_votes[point_idx], axis=-1))
            # And retrieve his most confident prediction
            max_vote[num_experts, point_idx] = local_hard_votes[point_idx, most_confident]
        
        # Some voting types are trivial and can be done outside the loop
        # Average voting
        average_confidence = np.mean(local_soft_votes, axis=1)
        average_vote[num_experts] = np.argmax(average_confidence, axis=-1)
        # Median voting
        median_confidence = np.median(local_soft_votes, axis=1)
        median_vote[num_experts] = np.argmax(median_confidence, axis=-1)
        
        # Compute accuracy vs. ground truth
        maj_dice_acc = [np.mean(maj_dice_vote[i] == np.squeeze(y_val)) for i in range(num_classes)]
        max_acc      = [np.mean(max_vote[i] == np.squeeze(y_val)) for i in range(num_classes)]
        average_acc  = [np.mean(average_vote[i] == np.squeeze(y_val)) for i in range(num_classes)]
        median_acc   = [np.mean(median_vote[i] == np.squeeze(y_val)) for i in range(num_classes)]
        
        # Plot and save to file
        plt.figure()
        plt.plot(range(1, num_classes+1), maj_dice_acc, label='Majority vote', marker='+')
        plt.plot(range(1, num_classes+1), max_acc, label='Max vote', marker='x')
        plt.plot(range(1, num_classes+1), average_acc, label='Average vote', marker='o')
        plt.plot(range(1, num_classes+1), median_acc, label='Median vote', marker='*')
        plt.xlabel('Number of experts')
        plt.ylabel('Validation accuracy')
        plt.legend()
        plt.grid()
        plt.savefig(local_dir + '/expert_mixture.png', dpi=600)
        plt.close()