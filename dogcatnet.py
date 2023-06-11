# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:02:50 2020

@author: singh
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import traceback
# K.common.set_image_data_format('channels_first')

class ConvNN():
    # Basic CNN model for classification of Dogs and Cats
    @staticmethod
    def build(width, height, channels, nb_class, bnEps=2e-5, bnMom=0.9, reg=0.0001):
        
        if K.image_data_format() == 'channels_first':
            # For Theano backend, the input shape should be "channels first",
            # and "bn_axis" should also be 1
            inputShape = (channels, width, height)
            bn_axis = 1
        else:
            # For Tensorflow backend, the input shape should be "channels last",
            # and "bn_axis" should also be -1
            inputShape = (width, height, channels)
            bn_axis = -1
        
        # Initializing the model
        model = Sequential()
        
        # Conv('relu') => BatchNorm => MaxPool
        model.add(Conv2D(8, (5, 5), padding='same', kernel_regularizer=l2(reg), input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=bn_axis, epsilon=bnEps, momentum=bnMom))
        model.add(MaxPooling2D(2, 2))
        
        # First set of [Conv('relu') => BatchNorm] * 2 => MaxPool
        model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=bn_axis, epsilon=bnEps, momentum=bnMom))
        model.add(Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=bn_axis, epsilon=bnEps, momentum=bnMom))
        model.add(MaxPooling2D(2, 2))
        
        # Second set of [Conv('relu') => BatchNorm] * 2 => MaxPool
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=bn_axis, epsilon=bnEps, momentum=bnMom))
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=bn_axis, epsilon=bnEps, momentum=bnMom))
        model.add(MaxPooling2D(2, 2))
        
        # Pair of FC('relu') => BatchNorm => Dropout
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        model.add(Dense(128))
        model.add(Activation('relu'))    
        model.add(BatchNormalization())
        model.add(Dropout(0.15))
        
        # Second set of FC('relu') => BatchNorm => Dropout
        # model.add(Flatten())
        # model.add(Dense(128, kernel_initializer='uniform'))
        # model.add(Activation('relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        
        # Sigmoid/Softmax Activation
        if nb_class == 1:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
        
        if nb_class > 1:
            model.add(Dense(nb_class))
            model.add(Activation('softmax'))
            
        if nb_class < 1:
            traceback.print_exc()
            
            
        # Return the constructed network
        return model
        
    @staticmethod
    def build_simple(image_dims: tuple):
        # To be used only with Tensorflow backend
        if len(image_dims) == 2:
            width, height = image_dims
        else: 
            raise ValueError("The tuple should only contain image width and height respectively") 
            
        inputShape = (width, height, 3)
        
        # Model construction
        model = Sequential()
        
        model.add(Conv2D(16, (3, 3), strides=1, padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(32, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
              
        model.add(Flatten())
        model.add(Dense(48))
        model.add(Activation('relu'))
        # model.add(Dropout(0.1))

        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        return model
    
    def build_soa(image_dims: tuple, nb_layers=1):
        # To be used only with tensorflow backend
        if len(image_dims) == 2:
            width, height = image_dims
        else:
            raise ValueError("Invalid Input")
        
        inputShape = (width, height, 3)
        
        nb_feat = 64
        
        model = Sequential()
        
        model.add(Conv2D(nb_feat, (3, 3), kernel_initializer='lecun_uniform', bias_initializer='ones', input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        
        for i in range(1, int(nb_layers) + 1):
            model.add(Conv2D(nb_feat * i, (3, 3), kernel_initializer='lecun_uniform', bias_initializer='ones'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        
        model.add(Flatten())
        model.add(Dense(nb_feat / 2 * (2 ** nb_layers)))
        model.add(Activation('relu'))
        model.add(Dense(2))
        model.add(Activation('softmax'))
        
        return model
        
        
        
        
        

