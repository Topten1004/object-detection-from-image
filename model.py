import numpy as np
import tensorflow as tf

from keras.layers import Conv2D, Conv2DTranspose, concatenate



def Conv(n_filters, filter_width):
    return Conv2D(n_filters, filter_width, 
                  strides=2, padding="same", activation="relu")

def Deconv(n_filters, filter_width):
    return Conv2DTranspose(n_filters, filter_width, 
                           strides=2, padding="same", activation="relu")

def Encoder_Decoder(inputs):
    d1 = Conv(128, (3,3))(inputs)
    d2 = Conv(128, (3,3))(d1)
    d3 = Conv(256, (3,3))(d2)
    d4 = Conv(512, (3,3))(d3)
    d5 = Conv(512, (3,3))(d4)    


    u1= Deconv(512, (3,3))(d5)
    u1 = concatenate([ u1, d4 ])
    u2 = Deconv(256, (3,3))(u1)
    u2 = concatenate([ u2, d3 ])
    u3 = Deconv(128, (3,3))(u2)
    u3 = concatenate([u3, d2])
    u4 = Deconv(128, (3,3))(u3)
    u4 = concatenate([u4, d1])
    u5 = Deconv(3, (3,3))(u4)
    u5 = concatenate([u5, inputs])    
    output = Conv2D(3,(2,2), strides=1, padding='same')(u5)
    return output

def AutoEncoder():
    X = tf.keras.Input(shape=(512, 512, 3))    
    X_pred = Encoder_Decoder(X)
    return tf.keras.Model(inputs=X, outputs=X_pred)

# autoencoder = AutoEncoder()
# autoencoder.compile( optimizer='adam', loss='mse', metrics=['accuracy'])
# autoencoder.summary()


