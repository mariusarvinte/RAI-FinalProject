# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:46:12 2019

@author: ma56473
"""

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Flatten, Reshape, concatenate
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import glorot_uniform
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import numpy as np
import os
import matplotlib

# NOTE: This will mess up your backend, it needs to be switched back to 'auto'
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ioff()

# Main object
class SupervisedAAE():
    def __init__(self, latent_dim, num_classes, hidden_dim, num_layers, 
                 filter_size, centroids, img_shape, global_seed, weight_seed,
                 centroid_seed, batch_seed, try_idx, arch):
        # Architecture parameters
        self.arch        = arch # This controls FC vs. Conv.
        self.latent_dim  = latent_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.filter_size = filter_size
        
        # Dataset parameters
        self.img_shape   = img_shape
        self.num_classes = num_classes
        self.input_dim   = np.prod(img_shape)
        
        # Latent distribution parameters
        self.centroids   = centroids
        # Covariance is always identity
        
        # Seeding parameters
        self.global_seed   = global_seed
        self.weight_seed   = weight_seed
        self.centroid_seed = centroid_seed
        self.batch_seed    = batch_seed
        
        # Optimizers
        self.optimizer_ae   = Adam(1e-4, amsgrad=True)
        self.optimizer_disc = Adam(1e-4, amsgrad=True)
        self.optimizer_clf  = Adam(1e-4)
        
        # Utilities
        self.try_idx = try_idx
        self.global_dirname = 'restart_results/global' + str(self.global_seed)
        self.dirname = self.global_dirname + \
                '/weight' + str(self.weight_seed) + '_try' + str(self.try_idx)
        # Create folder structure
        if not os.path.exists(self.global_dirname):
            os.makedirs(self.global_dirname)
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)
            
        # Instantiate model
        self.initModel()

    def genEncoder(self):
        encoder = Sequential()
        if self.arch == 'FC':
            # Flatten
            encoder.add(Flatten(input_shape=self.img_shape))
            # Input and hidden layers
            for layer_idx in range(self.num_layers):
                encoder.add(Dense(self.hidden_dim, activation='relu',
                                  kernel_initializer=glorot_uniform(seed=self.weight_seed+layer_idx)))
            # Final layer
            encoder.add(Dense(self.latent_dim,
                              kernel_initializer=glorot_uniform(seed=self.weight_seed+self.num_layers)))
        elif self.arch == 'Conv':
            # Input layer
            encoder.add(Conv2D(filters=self.hidden_dim, 
                               kernel_size=(self.filter_size, self.filter_size),
                               strides=(2, 2),
                               padding='same',
                               data_format='channels_last',
                               activation='relu',
                               kernel_initializer=glorot_uniform(seed=self.weight_seed),
                               input_shape=self.img_shape))
            # Hidden layers
            for layer_idx in range(self.num_layers):
                encoder.add(Conv2D(filters=self.hidden_dim, 
                               kernel_size=(self.filter_size, self.filter_size),
                               strides=(2, 2),
                               padding='same',
                               activation='relu',
                               kernel_initializer=glorot_uniform(seed=self.weight_seed+layer_idx+1),
                               data_format='channels_last'))
            # Final layer
            encoder.add(Conv2D(filters=self.latent_dim, 
                               kernel_size=(self.filter_size, self.filter_size),
                               strides=(2, 2),
                               padding='same',
                               kernel_initializer=glorot_uniform(seed=self.weight_seed+self.num_layers+1),
                               data_format='channels_last'))
            # Flatten
            encoder.add(Flatten())
        
        # Print summary
        print('The encoder architecture:')
        encoder.summary()
        
        return encoder

    def genDecoder(self):
        decoder = Sequential()
        if self.arch == 'FC':
            # Input layer
            decoder.add(Dense(self.hidden_dim, activation='relu', input_dim=self.latent_dim,
                              kernel_initializer=glorot_uniform(seed=self.weight_seed+20)))
            # Hidden layers
            for layer_idx in range(self.num_layers-1):
                decoder.add(Dense(self.hidden_dim, activation='relu',
                                  kernel_initializer=glorot_uniform(seed=self.weight_seed+21+layer_idx)))
            # Output layer
            decoder.add(Dense(np.prod(self.img_shape), activation='sigmoid',
                              kernel_initializer=glorot_uniform(seed=self.weight_seed+21+self.num_layers)))
            # Reshape back to image
            decoder.add(Reshape(self.img_shape))
        elif self.arch == 'Conv':
            # Reshape
            decoder.add(Reshape(target_shape=(1, 1, self.latent_dim),
                                input_shape=(self.latent_dim,)))
            # Input layer
            decoder.add(Conv2DTranspose(filters=self.hidden_dim,
                                        kernel_size=(self.filter_size, self.filter_size),
                                        strides=(2, 2),
                                        padding='same',
                                        activation='relu',
                                        # 20 is a pseudo-random offset
                                        kernel_initializer=glorot_uniform(seed=self.weight_seed+20),
                                        data_format='channels_last'))
            # Hidden layers
            for layer_idx in range(self.num_layers-2):
                decoder.add(Conv2DTranspose(filters=self.hidden_dim,
                                            kernel_size=(self.filter_size, self.filter_size),
                                            strides=(2, 2),
                                            padding='same',
                                            activation='relu',
                                            kernel_initializer=glorot_uniform(seed=self.weight_seed+21+layer_idx),
                                            data_format='channels_last'))
            # Second-to-final layer
            decoder.add(Conv2DTranspose(filters=self.hidden_dim,
                                        kernel_size=(self.filter_size, self.filter_size),
                                        strides=(2, 2),
                                        output_padding=(1, 1),
                                        padding='same',
                                        activation='relu',
                                        kernel_initializer=glorot_uniform(seed=self.weight_seed+21+self.num_layers),
                                        data_format='channels_last'))
            
            # First-to-final layer
            decoder.add(Conv2DTranspose(filters=self.hidden_dim,
                                        kernel_size=(self.filter_size, self.filter_size),
                                        strides=(2, 2),
                                        output_padding=(1, 1),
                                        padding='same',
                                        activation='relu',
                                        kernel_initializer=glorot_uniform(seed=self.weight_seed+22+self.num_layers),
                                        data_format='channels_last'))
            # Final layer
            decoder.add(Conv2DTranspose(filters=self.img_shape[-1],
                                        kernel_size=(self.filter_size, self.filter_size),
                                        strides=(2, 2),
                                        output_padding=(1, 1),
                                        padding='same',
                                        activation='sigmoid',
                                        kernel_initializer=glorot_uniform(seed=self.weight_seed+23+self.num_layers),
                                        data_format='channels_last'))
            
        # Print summary
        print('The decoder architecture:')
        decoder.summary()
        
        return decoder

    def genDiscriminator(self):
        # Latent and label inputs
        latent_input = Input(shape=(self.latent_dim,))
        labels_input = Input(shape=(self.num_classes,))
        concated = concatenate([latent_input, labels_input])
        
        discriminator = Sequential()
        # Input layer
        discriminator.add(Dense(self.hidden_dim, activation='relu',
                                input_dim=self.latent_dim+self.num_classes,
                                kernel_initializer=glorot_uniform(seed=self.weight_seed+60)))
        # Hidden layers
        for layer_idx in range(self.num_layers-1):
            discriminator.add(Dense(self.hidden_dim, activation='relu',
                                    kernel_initializer=glorot_uniform(seed=self.weight_seed+61+layer_idx)))
        # Output layer
        discriminator.add(Dense(1, activation='sigmoid',
                                kernel_initializer=glorot_uniform(seed=self.weight_seed+62+self.num_layers)))
        out = discriminator(concated)
        # Label-aware model
        discriminator_model = Model([latent_input, labels_input], out)
        
        # Print summary
        print('The discriminator architecture:')
        discriminator_model.summary()
        
        return discriminator_model

    def initModel(self):
        # Instantiate models
        self.encoder = self.genEncoder()
        self.decoder = self.genDecoder()
        self.discriminator = self.genDiscriminator()
        
        # Input image
        img = Input(shape=self.img_shape)
        # Input labels (to discriminator, during training only)
        label_code = Input(shape=(self.num_classes,))
        
        # Latent image
        encoded_repr = self.encoder(img)
        # Recovered image
        gen_img = self.decoder(encoded_repr)
        
        # AE model
        self.autoencoder = Model(img, gen_img)
        
        # Discriminator prediction
        valid = self.discriminator([encoded_repr, label_code])
        
        # Generator model
        self.encoder_discriminator = Model([img, label_code], valid)
        
        # Compile discriminator
        self.discriminator.compile(optimizer=self.optimizer_disc,
                                   loss='binary_crossentropy',
                                   metrics=['accuracy'])
        # Compile autoencoder
        self.autoencoder.compile(optimizer=self.optimizer_ae,
                                 loss ='mse')
        
        # Compile generator        
        self.encoder_discriminator.compile(optimizer=self.optimizer_disc,
                                           loss='binary_crossentropy',
                                           metrics=['accuracy'])
    
    def saveWeights(self, epoch):
        # Save autoencoder weights
        self.autoencoder.save_weights(self.dirname + '/ae_weights_epoch' + str(epoch) + '.h5')
        # Save discriminator weights
        self.discriminator.save_weights(self.dirname + '/disc_weights_epoch' + str(epoch) + '.h5')
    
    def saveLogs(self, epoch, x_test, y_test):
        # Open figure
        fig = plt.figure(figsize=[20, 20])
        
        # Compute latent representation of fresh data
        lat = self.encoder.predict(x_test)
        ax = fig.add_subplot(1, 1, 1)
        
        # Decode a small number of reconstructed images
        dec = self.decoder.predict(lat[:20])
        
        # Plot and save to file
        n_cols = 10
        n_rows = 2
        g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False)
        for i in range(n_rows):
            for j in range(n_cols):
                b[i][j].imshow(np.squeeze(dec[i*n_cols+j]),
                 cmap='gray',
                 interpolation='none')
                b[i][j].axis('off')
        plt.savefig(self.dirname + '/decoded_images_' + str(epoch) + 'epochs.png',
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()
                
        # Compute and print silhouette score
        s_score = silhouette_score(lat, y_test, random_state=self.weight_seed)
        print('Silhouette score is: ' + str(s_score))
        # Write to global file
        with open(self.global_dirname + '/silhouettes.txt', 'a+') as file:
            file.write('Try number %d, Weight seed %d, Epoch %d, Silhouette score: %f\n' % (self.try_idx,
                                                                        self.weight_seed, epoch, s_score))
        
        # TSNE for more than 2-dimensions
        if self.latent_dim > 2:
            lat_embedded = TSNE(n_iter=600, verbose=2, random_state=self.weight_seed).fit_transform(lat)
            ax.scatter(lat_embedded[:, 0], lat_embedded[:, 1], c=y_test)
        else:
            ax.scatter(lat[:, 0], lat[:, 1], c=y_test)                  
        
        fig.savefig(self.dirname + '/latent_map_'+ str(epoch) + 'epochs.png')

    def train(self, x_train, y_train, x_test, y_test, batch_size, epochs, save_interval):        
        # Image augmentation object
        datagen = ImageDataGenerator(rotation_range=0, # Rotation in angles
                width_shift_range=0.,
                height_shift_range=0.,
                shear_range=0., # Image shearing, counter-clockwise
                horizontal_flip=False, # TODO: These may mess up the training
                vertical_flip=False,
                fill_mode='nearest')
        # Fit to data
        datagen.fit(x_train, seed=self.weight_seed)
        
        # Main loop
        for epoch in range(epochs):
            # Counter
            batch_idx = 0
            for imgs, y_batch in datagen.flow(x_train, y_train, shuffle=False,
                                              batch_size=batch_size, seed=(self.batch_seed + epoch)):
                # Counter
                batch_idx = batch_idx + 1
                # Generate a half batch of new images
                latent_fake = self.encoder.predict(imgs)
    
                # Generate random samples
                (latent_real, labels) = self.generateRandomVectors(y_batch, seed=(self.batch_seed+epoch+batch_idx))
                valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
                
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([latent_real, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([latent_fake, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
                # Generator wants the discriminator to label the generated representations as valid
                valid_y = np.ones((batch_size, 1))
    
                # Train autoencoder for reconstruction
                g_loss_reconstruction = self.autoencoder.train_on_batch(imgs, imgs)
    
                # Train generator
                g_logg_similarity = self.encoder_discriminator.train_on_batch([imgs, labels], valid_y)
                
                # Plot progress per batch
                print("Epoch %d, batch %d : [D loss: %f, acc: %.2f%%] [G acc: %f, mse: %f]" % (epoch,
                       batch_idx, d_loss[0], 100*d_loss[1],
                       g_logg_similarity[1], g_loss_reconstruction))
                
                # Break loop by hand (Keras trick)
                if batch_idx >= len(x_train) / batch_size:
                    break
                
            # Write to file
            if (epoch % save_interval == 0):
                self.saveWeights(epoch)
                self.saveLogs(epoch, x_test, y_test)

    def generateRandomVectors(self, y_train, seed):
        # Seed
        np.random.seed(seed)
        
        # This is our sampled latent space
        vectors = []
        labels = np.zeros((len(y_train), self.num_classes))
        labels[range(len(y_train)), np.array(y_train).astype(int)] = 1
        
        for index, y in enumerate(y_train):            
            # Impose the desired latent space distribution
            mean = 10*self.centroids[y]
            # Unit covariance
            cov  = 3*np.eye(self.latent_dim)
            # Draw random value
            vec = np.random.multivariate_normal(mean=mean, cov=cov, size=1)
            vectors.append(vec)
            
        return (np.array(vectors).reshape(-1, self.latent_dim), labels)