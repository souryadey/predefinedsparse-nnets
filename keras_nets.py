#==============================================================================
# Pre-defined nets for Keras
# Each returns a Keras model and an optimizer function
# Sourya Dey, USC
#==============================================================================

#==============================================================================
#==============================================================================
# # Imports
#==============================================================================
#==============================================================================
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
#==============================================================================


#==============================================================================
#==============================================================================
# # General CL functions
#==============================================================================
#==============================================================================
def any_cl_only(config, activation='relu', output_activation='softmax', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=l2(0.)):
    '''
    Any MLP network
    lr and decay are set to defaults for Adam as in Keras
    '''
    model = Sequential()
    model = add_cls(model, config, activation=activation, output_activation=output_activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer)
    return model


def add_cls(model, config, activation='relu', output_activation='softmax', kernel_initializer='he_normal', bias_initializer='zeros', kernel_regularizer=l2(0.)):
    '''
    Helper function to add CLs to any EXISTING model
    Inputs:
        model: Existing model
        config: Must know what the exact shape of CL portion is
            Eg: If adding CLs to a CIFAR net with CNNs, then we must know what the number of neurons after flattening is
            Eg: cifar_gitblog1_example gives 4096 neurons after flattening, so maybe config = np.array([4096,512,10])
        kernel_initializer, bias_initializer, kernel_regularizer: Use the same for all layers
        Use activation for all hidden layers, output_activation for output layer
    Output:
        Model with CLs attached
    Possible improvement:
        Add dropout as as ndarray input with size = len(config)-1
    '''
    for i in range(1,len(config)):
        if i==len(config)-1: #Use output_activation for output layer and name it 'output'
            model.add(Dense(config[i], input_shape=(config[i-1],), activation=output_activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, name='output'))
        else: #Standard hidden layers
            model.add(Dense(config[i], input_shape=(config[i-1],), activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))
    return model
#==============================================================================



def cifar_deep(config=np.array([4096,512,10]), activation='relu', output_activation='softmax', kernel_initializer='he_normal',
                           bias_initializer='zeros', kernel_regularizer=l2(0.), dropout=0.5):
    '''
    dropout: Fraction of units to DROP, i.e. set to 0. for no dropout
    '''
    model = Sequential()
    model.add(Conv2D(60, (3, 3), padding='same', input_shape=(32,32,3))) #note that conv layers have no regularizer, but they do have dropout
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(Conv2D(60, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(125, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(Conv2D(125, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(250, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(Conv2D(250, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten(name='flatten_before_mlp'))
    model = add_cls(model, config, activation=activation, output_activation=output_activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer)
#    NOTE: Original code uses dropout(0.5) in between any 2 CLs
#==============================================================================
#     opt = SGD(lr=lr,decay=decay,momentum=0.9)
#     opt = Adam(lr=lr, decay=decay)
#==============================================================================
    return model


def cifar_shallow(config=np.array([4000,500,100]), activation='relu', output_activation='softmax', kernel_initializer='he_normal',
                           bias_initializer='zeros', kernel_regularizer=l2(0.), dropout=0.25):
    '''
    Intentionally make it harder for MLP by reducing the amount of feature extraction via CNN
    '''
    model = Sequential()
    model.add(Conv2D(250, (5, 5), padding='same', input_shape=(32,32,3)))
    model.add(BatchNormalization(axis=3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(dropout))

    model.add(Flatten(name='flatten_before_mlp'))
    model = add_cls(model, config, activation=activation, output_activation=output_activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer)
    return model
