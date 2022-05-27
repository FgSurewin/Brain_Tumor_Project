
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Conv2DTranspose, Concatenate, Activation, Dropout, Flatten, Dense, Input,concatenate, UpSampling2D
from keras.callbacks import EarlyStopping
import tensorflow as tf

def convolutional_layer(input, filter_nums):
    output = Conv2D(filter_nums, 3, padding="same", kernel_initializer = 'he_normal')(input)
    output = Activation("relu")(output)

    output = Conv2D(filter_nums, 3, padding="same", kernel_initializer = 'he_normal')(output)
    output = Activation("relu")(output)
    return output


def encoder_layer(input, filter_nums, has_drop_out=False, drop_value=0.5):
    output = convolutional_layer(input, filter_nums)
    if has_drop_out: 
      drop = Dropout(drop_value)(output)
      pool = MaxPooling2D((2, 2))(drop)
      return output, pool
    else:
      pool = MaxPooling2D((2, 2))(output)
      return output, pool 
    

def decoder_layer(input, conte_feature, filter_nums):
    conv_tran = Conv2DTranspose(filter_nums, (2, 2), strides=2, padding="same", kernel_initializer = 'he_normal')(input)
    uconv = Concatenate()([conv_tran, conte_feature])
    output = convolutional_layer(uconv, filter_nums)
    return output


INPUT_SHAPE = (256, 256, 3)
def build_unet():
    inputs = Input(INPUT_SHAPE)
    
    # Encoder process
    con1, pool1 = encoder_layer(inputs, 64)
    con2, pool2 = encoder_layer(pool1, 128)
    con3, pool3 = encoder_layer(pool2, 256)
    con4, pool4 = encoder_layer(pool3, 512, True)

    # Bridge/Middle
    bridge = convolutional_layer(pool4, 1024) 
    drop5 = Dropout(0.5)(bridge)

    # Decoder process
    dec1 = decoder_layer(drop5, con4, 512)
    dec2 = decoder_layer(dec1, con3, 256)
    dec3 = decoder_layer(dec2, con2, 128)
    dec4 = decoder_layer(dec3, con1, 64)

    outputs = Conv2D(1, 1, padding="same", activation='sigmoid')(dec4) 
    model = Model(inputs, outputs, name="U-Net")
    return model