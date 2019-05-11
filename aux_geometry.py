# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:47:14 2019

@author: ma56473
"""

import numpy as np
import scipy

# Find centroids (snapped to nearest data point)
def find_data_centroid(x_data, y_data, n_clusters, n_dims):
    sort_index = np.squeeze(np.argsort(y_data, axis=0))

    x_data = x_data[sort_index]
    y_data = y_data[sort_index]

    avg_pts = np.zeros((n_clusters, n_dims))

    # find closest data pt to each fake center for each class
    for i in range(n_clusters):
        x_point = x_data[np.argwhere(y_data == i)[:,0], :]
        avg_pts[i,:] = np.average(x_point, axis=0)
    return avg_pts

# Auxiliary function to generate centroids in n-D
def gen_centroids(latent_dim, num_classes, seed, num_tries):
    # Record closest pair of points
    centroids = np.zeros((num_classes, latent_dim))
    min_dist  = 0
    # Seed
    np.random.seed(seed)
    
    # Pick random points in [-1, 1]
    # Pick most-spread apart points after a fixed number of tries
    for try_idx in range(num_tries):
        new_centroids = np.random.uniform(low=-1., high=1., size=((num_classes, latent_dim)))
        # Compute minimum pairwise distance
        new_dist = scipy.spatial.distance.cdist(new_centroids, new_centroids)
        np.fill_diagonal(new_dist, np.inf)
        new_dist = np.min(new_dist)
        # Pick the best max-min solution
        if new_dist > min_dist:
            min_dist  = new_dist
            centroids = new_centroids
    
    return centroids

# Generate noisy clusters around centroids
def gen_synthetic_data(latent_dim, num_classes, centroids, num_points,
                       noise_mean, noise_sigma):
    # Outputs
    X = []
    y = []
    
    # Loop
    for i in range(num_classes):
    	for j in range(num_points):
    		X.append(centroids[i,:] + np.random.normal(noise_mean, noise_sigma, (1, latent_dim)))
    		y.append(i)
    
    # Reshape
    X = np.squeeze(np.asarray(X))
    y = np.expand_dims(np.asarray(y), axis=-1)
    # Shuffle
    random_permutation = np.random.permutation(num_classes*num_points)
    X = X[random_permutation]
    y = y[random_permutation]
    
    # Training/validation split
    x_train, x_val = np.split(X, [int(0.8*np.shape(X)[0])])
    y_train, y_val = np.split(y, [int(0.8*np.shape(y)[0])])
    
    return x_train, x_val, y_train, y_val