import numpy as np



from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
import tensorflow.keras.losses as losses

import tensorflow.keras.initializers as initializers

import tensorflow as tf
from tensorflow import keras
# from keras.layers import Add, Dense, Dropout, Input, Lambda, concatenate
# from keras import metrics, regularizers, initializers
# from keras.optimizers import Adam, Adamax


def createCNNModel(data_shape, num_classes, seed):
    # Trying a new initializer to help the network start actually learning
    
    cnn_model = Sequential([
        # Reshaping the data, to add an extra dimension so the Conv2D layers will work with it
        # layers.Reshape((data_shape, 1), input_shape=data_shape),
        
        layers.Rescaling(1./255, input_shape=data_shape),
        layers.Conv1D(filters=16, kernel_size=7, strides=(1,), 
                      padding='same', activation='relu',
                      kernel_initializer=initializers.GlorotNormal(seed+0)),
        layers.MaxPooling1D(),
        layers.Dropout(0.2),
        layers.Conv1D(filters=32, kernel_size=5, strides=(1,), 
                      padding='same', activation='relu',
                      kernel_initializer=initializers.GlorotNormal(seed+1)),
        layers.MaxPooling1D(),
        layers.Dropout(0.2),
        layers.Conv1D(filters=64, kernel_size=3, strides=(1,), 
                      padding='same', activation='relu',
                      kernel_initializer=initializers.GlorotNormal(seed+2)),
        layers.MaxPooling1D(),
        layers.Dropout(0.2),
        layers.Conv1D(filters=128, kernel_size=3, strides=(1,), 
                      padding='same', activation='relu',
                      kernel_initializer=initializers.GlorotNormal(seed+2)),
        layers.MaxPooling1D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer=initializers.GlorotNormal(seed+3)),
        layers.Dense(num_classes, activation='softmax', kernel_initializer=initializers.GlorotNormal(seed+4))
    ])
    return cnn_model

def initializeCNN(input_shape, num_classes, lr=1e-1, seed=10):
    cnn_model = createCNNModel(input_shape, num_classes, seed)
    
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=5000,
            initial_learning_rate=lr,
            decay_rate=0.95
            )
    cnn_optimizer = keras.optimizers.Adam(
            #learning_rate=lr_scheduler,
            learning_rate=lr,
            amsgrad=False
            )

    cnn_model.compile(optimizer=cnn_optimizer, 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

    return(cnn_model)


def createCNNModel_withAug(data_shape, num_classes):

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    cnn_model = Sequential([
        # Reshaping the data, to add an extra dimension so the Conv2D layers will work with it
        # layers.Reshape((data_shape, 1), input_shape=data_shape),
        data_augmentation,
        layers.Rescaling(1./255, input_shape=data_shape),
        layers.Conv2D(filters=16, kernel_size=3, strides=(1,1), 
                      padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), 
                      padding='same', activation='relu'),
        layers.MaxPooling2D(), 
        layers.Conv2D(filters=64, kernel_size=3, strides=(1,1), 
                      padding='same', activation='relu'),
        layers.MaxPooling2D(), 
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return(cnn_model)

def initializeCNN_withAug(input_shape, num_classes, lr=5e-1):
    cnn_model = createCNNModel_withAug(input_shape, num_classes)
    
    # Setting up a dynamic learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=500,
            initial_learning_rate=lr,
            decay_rate=0.99
            )
    cnn_optimizer = keras.optimizers.Adam(
            #learning_rate=lr_scheduler,
            learning_rate=lr,
            amsgrad=False
            )

    cnn_model.compile(optimizer=cnn_optimizer, 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

    return(cnn_model)



def initialize_MobileNet(input_shape, num_classes, lr=5e-1):
    
    mobileNet_inShape = (input_shape[0], input_shape[1], 3)


    base_model = tf.keras.applications.MobileNetV2(input_shape=mobileNet_inShape, 
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False # Freezing the convolutional layers since we are 
                                 # are just training a layer on the output features


    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropoutLayer = tf.keras.layers.Dropout(0.2)
     
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    
    cnn_model = Sequential([
            base_model,
            global_average_layer,
            dropoutLayer,
            prediction_layer
        ])
    

    # Setting up a dynamic learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=5000,
            initial_learning_rate=lr,
            decay_rate=0.9
            )
    cnn_optimizer = keras.optimizers.Adam(
            #learning_rate=lr_scheduler,
            learning_rate=lr,
            amsgrad=False
            )
    
    cnn_model.compile(optimizer=cnn_optimizer, 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

    return(cnn_model)

def initialize_MobileNet_2Deep(input_shape, num_classes, lr=5e-1):
    
    mobileNet_inShape = (input_shape[0], input_shape[1], 3)


    base_model = tf.keras.applications.MobileNetV2(input_shape=mobileNet_inShape, 
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False # Freezing the convolutional layers since we are 
                                 # are just training a layer on the output features


    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropoutLayer = tf.keras.layers.Dropout(0.2)
    fullyConnectedLayer1 = tf.keras.layers.Dense(50, activation='relu')
     
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    
    cnn_model = Sequential([
            base_model,
            global_average_layer,
            dropoutLayer,
            fullyConnectedLayer1,
            prediction_layer
        ])
    

    # Setting up a dynamic learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=5000,
            initial_learning_rate=lr,
            decay_rate=0.9
            )
    cnn_optimizer = keras.optimizers.Adam(
            #learning_rate=lr_scheduler,
            learning_rate=lr,
            amsgrad=False
            )
    
    cnn_model.compile(optimizer=cnn_optimizer, 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

    return(cnn_model)


def initialize_MobileNetV3(input_shape, num_classes, lr=5e-1):
    mobileNetV3_inShape = (input_shape[0], input_shape[1], 3)

    base_model = tf.keras.applications.MobileNetV3Large(input_shape=mobileNetV3_inShape, 
                                                   include_top=False, # Not including the classifier
                                                                      # portion of the CNN
                                                   weights='imagenet')

    base_model.trainable = False # Freezing the convolutional layers since we are 
                                 # are just training a layer on the output features

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropoutLayer = tf.keras.layers.Dropout(0.2)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    
    cnn_model = Sequential([
            base_model,
            global_average_layer,
            dropoutLayer,
            prediction_layer
        ])
    

    # Setting up a dynamic learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=5000,
            initial_learning_rate=lr,
            decay_rate=0.9
            )
    cnn_optimizer = keras.optimizers.Adam(
#            learning_rate=lr_scheduler,
            learning_rate=lr,
            amsgrad=False
            )
    
    cnn_model.compile(optimizer=cnn_optimizer, 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

    return(cnn_model)



def initialize_ConvNeXt(input_shape, num_classes, lr=5e-1):
    ConvNeXt_inShape = (input_shape[0], input_shape[1], 3)

    base_model = tf.keras.applications.convnext.ConvNeXtBase(input_shape=ConvNeXt_inShape, 
                                                   include_top=False, # Not including the classifier
                                                                      # portion of the CNN
                                                   weights='imagenet')

    base_model.trainable = False # Freezing the convolutional layers since we are 
                                 # are just training a layer on the output features

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropoutLayer = tf.keras.layers.Dropout(0.2)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    
    cnn_model = Sequential([
            base_model,
            global_average_layer,
            dropoutLayer,
            prediction_layer
        ])
    

    # Setting up a dynamic learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=5000,
            initial_learning_rate=lr,
            decay_rate=0.9
            )
    cnn_optimizer = keras.optimizers.Adam(
#             learning_rate=lr_scheduler,
            learning_rate=lr,
            amsgrad=False
            )
    
    cnn_model.compile(optimizer=cnn_optimizer, 
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

    return(cnn_model)



# Code for the inception net taken from: https://ai.plainenglish.io/googlenet-inceptionv1-with-tensorflow-9e7f3a161e87
def inception(x,
              filters_1x1,
              filters_3x3_reduce,
              filters_3x3,
              filters_5x5_reduce,
              filters_5x5,
              filters_pool):
    path1 = layers.Conv2D(filters_1x1, (1, 1), padding='same',    activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(filters_3x3, (1, 1), padding='same', activation='relu')(path2)
    path3 = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(filters_5x5, (1, 1), padding='same', activation='relu')(path3)
    path4 = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(filters_pool, (1, 1), padding='same', activation='relu')(path4)
    return tf.concat([path1, path2, path3, path4], axis=3)


def InceptionNetModel(input_shape, out_shape):
    inp = layers.Input(shape=input_shape)

    input_tensor = layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=input_shape[1:])(inp)

    x = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling2D(3, strides=2)(x)

    x = layers.Conv2D(64, 1, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(192, 3, strides=1, padding='same', activation='relu')(x)

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = inception(x,
                filters_1x1=64,
                filters_3x3_reduce=96,
                filters_3x3=128,
                filters_5x5_reduce=16,
                filters_5x5=32,
                filters_pool=32)

    x = inception(x,
                filters_1x1=128,
                filters_3x3_reduce=128,
                filters_3x3=192,
                filters_5x5_reduce=32,
                filters_5x5=96,
                filters_pool=64)

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = inception(x,
                filters_1x1=192,
                filters_3x3_reduce=96,
                filters_3x3=208,
                filters_5x5_reduce=16,
                filters_5x5=48,
                filters_pool=64)

    aux1 = layers.AveragePooling2D((5, 5), strides=3)(x)
    aux1 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux1)
    aux1 = layers.Flatten()(aux1)
    aux1 = layers.Dense(1024, activation='relu')(aux1)
    aux1 = layers.Dropout(0.7)(aux1)
    aux1 = layers.Dense(10, activation='softmax')(aux1)

    x = inception(x,
                filters_1x1=160,
                filters_3x3_reduce=112,
                filters_3x3=224,
                filters_5x5_reduce=24,
                filters_5x5=64,
                filters_pool=64)

    x = inception(x,
                filters_1x1=128,
                filters_3x3_reduce=128,
                filters_3x3=256,
                filters_5x5_reduce=24,
                filters_5x5=64,
                filters_pool=64)

    x = inception(x,
                filters_1x1=112,
                filters_3x3_reduce=144,
                filters_3x3=288,
                filters_5x5_reduce=32,
                filters_5x5=64,
                filters_pool=64)

    aux2 = layers.AveragePooling2D((5, 5), strides=3)(x)
    aux2 = layers.Conv2D(128, 1, padding='same', activation='relu')(aux2)
    aux2 = layers.Flatten()(aux2)
    aux2 = layers.Dense(1024, activation='relu')(aux2)
    aux2 = layers.Dropout(0.7)(aux2)
    aux2 = layers.Dense(10, activation='softmax')(aux2)

    x = inception(x,
                filters_1x1=256,
                filters_3x3_reduce=160,
                filters_3x3=320,
                filters_5x5_reduce=32,
                filters_5x5=128,
                filters_pool=128)

    x = layers.MaxPooling2D(3, strides=2)(x)

    x = inception(x,
                filters_1x1=256,
                filters_3x3_reduce=160,
                filters_3x3=320,
                filters_5x5_reduce=32,
                filters_5x5=128,
                filters_pool=128)

    x = inception(x,
                  filters_1x1=384,
                  filters_3x3_reduce=192,
                  filters_3x3=384,
                  filters_5x5_reduce=48,
                  filters_5x5=128,
                  filters_pool=128)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.4)(x)
    out = layers.Dense(out_shape, activation='softmax')(x)
    
    # Making the model
    model = Model(inputs = inp, outputs = [out, aux1, aux2])

    return(model)
    
def initialize_InceptionNet(input_shape, num_classes, lr=1e-1):
    cnn_model = InceptionNetModel((input_shape[0], input_shape[1], 3), num_classes)    

    # Setting up a dynamic learning rate
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
            decay_steps=500,
            initial_learning_rate=lr,
            decay_rate=0.9
            )
    cnn_optimizer = keras.optimizers.Adam(
            learning_rate=lr_scheduler,
            amsgrad=False
            )
    
#    cnn_model.compile(optimizer=cnn_optimizer, 
#                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#                       metrics=['accuracy'])

    cnn_model.compile(optimizer=cnn_optimizer, 
            loss = [losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False),
                losses.SparseCategoricalCrossentropy(from_logits=False)],
            loss_weights=[1, 0.3, 0.3],
            metrics=['accuracy'])

            

    return(cnn_model)

