import os
import sys
import cv2
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from data_manager import DataManager

import matplotlib.pyplot as plt 

from model import AutoEncoder

from absl import app
from absl import flags

flags.DEFINE_integer("sample_size", 10, "samples to test")
flags.DEFINE_string("model", "./trained_model/DAE-model-timestamp.h5", "Path to a trained model (.h5 file)")
flags.DEFINE_boolean("use_noise", True, "sample noisey images")
FLAGS = flags.FLAGS


def sample(model, n_samples):
    
    manager = DataManager()
    outputpath = './celeba_output/'
    # _, X = manager.get_batch(n_samples, use_noise=FLAGS.use_noise)
    # X_pred = model.predict(X)

    X, imgName, height, width = manager.get_batch_test( n_samples, use_noise=False )    
    X_pred = model.predict( X )
    for i in range(len(imgName)):        
        output = cv2.resize(X_pred[i], (height[i]*2, width[i]*2))    
        # output = cv2.resize(X_pred[i], (256, 256))    
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)        
        cv2.imwrite(outputpath+imgName[i], output*255)
        # plt.imshow(output)
        # plt.show()

def load_model():
    """Set up and return the model."""
    model_path = os.path.abspath(FLAGS.model)    
    # model = tf.keras.models.load_model(model_path)

    model = AutoEncoder()
    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = tf.keras.losses.MSE

    # load most recent weights if model_path exists 
    if os.path.isfile(model_path):
        print("Loading model from", model_path)
        model.load_weights(model_path)

    model.compile(optimizer, loss, metrics=['acc'])
    model.summary()


    # # holds dimensions of latent vector once we find it
    # z_dim = None

    # # define encoder
    # encoder_in  = tf.keras.Input(shape=(32, 32, 1))
    # encoder_out = Encoder(encoder_in)
    # encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    # # load encoder weights and get the dimensions of the latent vector
    # for i, layer in enumerate(model.layers):
    #     encoder.layers[i] = layer
    #     if layer.name == "encoder_output":
    #         z_dim = (layer.get_weights()[0].shape[-1])
    #         break

    # # define encoder
    # decoder_in  = tf.keras.Input(shape=(z_dim,))
    # decoder_out = Decoder(decoder_in)
    # decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

    # # load decoder weights
    # found_decoder_weights = False
    # decoder_layer_cnt = 0
    # for i, layer in enumerate(model.layers):
    #     print(layer.name)
    #     weights = layer.get_weights()
    #     if len(layer.get_weights()) > 0:
    #         print(weights[0].shape, weights[1].shape)
    #     if "decoder_input" == layer.name:
    #         found_decoder_weights = True
    #     if found_decoder_weights:
    #         decoder_layer_cnt += 1
    #         print("dec:" + decoder.layers[decoder_layer_cnt].name)
    #         decoder.layers[decoder_layer_cnt].set_weights(weights)

    # encoder.summary()
    # decoder.summary()

    return model
       
def main(argv):
    if FLAGS.model == None:
        print("Please specify a path to a model with the --model flag")
        sys.exit()
    autoencoder = load_model()        
    sample(autoencoder, FLAGS.sample_size)

if __name__ == '__main__':
    app.run(main)
