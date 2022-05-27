from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf



learning_rate = 0.0005

IMAGE_SIZE = (256, 256, 3)

def make_model(transfer_model, learning_rate=learning_rate):
  model = transfer_model.output
  model = tf.keras.layers.Flatten()(model)
  model = tf.keras.layers.Dropout(rate=0.2)(model)
  model = tf.keras.layers.Dense(16, activation='relu')(model)
  model = tf.keras.layers.Dense(32, activation='relu')(model)
  model = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer='l1_l2')(model)
  model = tf.keras.models.Model(inputs=transfer_model.input, outputs = model)

  opt = tf.keras.optimizers.Adam(learning_rate= learning_rate, amsgrad=True)
  metrics = ['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

  model.compile(
          optimizer=opt, 
          loss="binary_crossentropy", 
          metrics=metrics,
      )

  return model

# patient early stopping
es_transfer_learning = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)