# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:49:28 2019

@author: ma56473
"""

from keras import backend as K
from keras.layers import Layer

# Custom loss function that takes multi-tensor input
# Uses the function-in-function trick to bypass Keras restrictions
def commitment_crossentropy(r1, r2, lambda_0, lambda_1, lambda_2):
    # Core function
    def loss(y_true, y_pred):
        return lambda_0 * K.binary_crossentropy(y_true, y_pred) + lambda_1 * r1 + lambda_2 * r2
    
    # Return function
    return loss

# Restricted commitment loss (using only R1)
def r1_crossentropy(r1, lambda_1):
    # Core functioon
    def loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + lambda_1 * r1
    
    # Return function
    return loss
    
# Trainable prototype layer with cosine-distance embedding
class CosineEmbedding(Layer):
    def __init__(self, num_vectors, latent_dim, **kwargs):
        self.num_vectors   = num_vectors
        self.latent_dim    = latent_dim
        
        super(CosineEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable p vectors
        self.trainable_p = self.add_weight(name='trainable_p',
                                           shape=(self.num_vectors, self.latent_dim),
                                           initializer='glorot_uniform',
                                           trainable=True)

        super(CosineEmbedding, self).build(input_shape)
    
    # Main functionality goes here
    def call(self, x):        
        # Cosine similarity via normalized inner products
        # Normalize batch
        norm_x = K.l2_normalize(x, axis=-1)
        # Normalize p vectors
        norm_trainable_p = K.l2_normalize(self.trainable_p, axis=-1)
        # Compute similarities
        trainable_dist = K.dot(norm_x, K.transpose(norm_trainable_p))

        # Concatenated output
        distances = trainable_dist
        
        # If similarity, output negative max instead
        # R1 cost function (min over batch, sum over p)
        r1_cost = -K.mean(K.max(distances, axis=0), axis=-1)
        
        # R2 cost function (min over p, sum over batch)
        r2_cost = -K.mean(K.max(distances, axis=-1), axis=-1)
      
        # Return triplet
        return [distances, r1_cost, r2_cost]
    
    def compute_output_shape(self, input_shape):
        # Always returns scalars for the two extra terms
        return [(input_shape[0], self.num_vectors), (1,), (1,)]
    
# Trainable prototype layer with Euclidean distance embedding
class EuclideanEmbedding(Layer):
    def __init__(self, num_vectors, latent_dim, **kwargs):
        self.num_vectors   = num_vectors
        self.latent_dim    = latent_dim
        
        super(EuclideanEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable p vectors
        self.trainable_p = self.add_weight(name='trainable_p',
                                           shape=(self.num_vectors, self.latent_dim),
                                           initializer='glorot_uniform',
                                           trainable=True)

        super(EuclideanEmbedding, self).build(input_shape)
    
    # Main functionality goes here
    def call(self, x):        
        # Use axis expansion on x for fast computation
        x_dim = K.expand_dims(x, axis=1)
        # Distance to trainable p vectors
        trainable_dist = K.sqrt(K.sum(K.square(x_dim - self.trainable_p), axis=-1))

        # Concatenated output
        distances = trainable_dist
        
        # R1 cost function (min over batch, sum over p)
        r1_cost = K.mean(K.min(distances, axis=0), axis=-1)
        
        # R2 cost function (min over p, sum over batch)
        r2_cost = K.mean(K.min(distances, axis=-1), axis=-1)
      
        # Return triplet
        return [distances, r1_cost, r2_cost]
    
    def compute_output_shape(self, input_shape):
        # Always returns scalars for the two extra terms
        return [(input_shape[0], self.num_vectors), (1,), (1,)]
    
# Trainable prototype layer with Euclidean distance embedding
class L1Embedding(Layer):
    def __init__(self, num_vectors, latent_dim, **kwargs):
        self.num_vectors   = num_vectors
        self.latent_dim    = latent_dim
        
        super(L1Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        # Trainable p vectors
        self.trainable_p = self.add_weight(name='trainable_p',
                                           shape=(self.num_vectors, self.latent_dim),
                                           initializer='glorot_uniform',
                                           trainable=True)

        super(L1Embedding, self).build(input_shape)
    
    # Main functionality goes here
    def call(self, x):        
        # Use axis expansion on x for fast computation
        x_dim = K.expand_dims(x, axis=1)
        # Distance to trainable p vectors
        trainable_dist = K.sum(K.abs(x_dim - self.trainable_p), axis=-1)

        # Concatenated output
        distances = trainable_dist
        
        # R1 cost function (min over batch, sum over p)
        r1_cost = K.mean(K.min(distances, axis=0), axis=-1)
        
        # R2 cost function (min over p, sum over batch)
        r2_cost = K.mean(K.min(distances, axis=-1), axis=-1)
      
        # Return triplet
        return [distances, r1_cost, r2_cost]
    
    def compute_output_shape(self, input_shape):
        # Always returns scalars for the two extra terms
        return [(input_shape[0], self.num_vectors), (1,), (1,)]