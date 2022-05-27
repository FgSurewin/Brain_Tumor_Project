from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
import tensorflow as tf




metrics = ['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

GRAY_INPUT_SHAPE = (256, 256, 1)
COLOR_INPUT_SHAPE = (256, 256, 3)

def model_builder(optimizer):
    INPUT_SHAPE = GRAY_INPUT_SHAPE   #change to (SIZE, SIZE, 3)


    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform', padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform', padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid')) 

    model.compile(
        optimizer=optimizer, 
        loss="binary_crossentropy", 
        metrics=metrics,
    )

    return model