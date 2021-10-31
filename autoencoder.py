import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from typing import List, Tuple, Optional

"""
Dense transpose layer from dense layer
"""
class DenseTranspose(tf.keras.layers.Layer):
    def __init__(self, dense, activation = None, **kwargs):
        self.dense = dense 
        self.activation = tf.keras.activations.get(activation)
        super(DenseTranspose, self).__init__(**kwargs)
    def build(self, batch_input_shape):
        self.b = self.add_weight(name= "bias", shape = [ self.dense.input_shape[-1]], initializer = "zeros")
        self.w = self.dense.weights[0]
        super().build(batch_input_shape)
        
        
    def call(self, inputs):
        z = tf.linalg.matmul(inputs, self.w, transpose_b = True)
        return self.activation(z + self.b)
    
    def get_weights(self):
        return {"w": np.shape(tf.transpose(self.w))}    
    @property 
    def weights_transpose(self):
        return tf.transpose(self.dense.weights[0])

"""
Encoder

isFirstInputLayer - use input dimension only in Input Layer
"""
class Encoder(tf.keras.layers.Layer):
    def __init__(self, widths : List[int] = [32,24], name : Optional[str] = "encoder", isFirstInputLayer : Optional[bool] = True, **kwargs):
        super(Encoder, self).__init__(name = name, **kwargs)
        self.latent_dim = widths[-1]
        self.input_dim = widths[0]
        self.encoder_layers = []
        for layer_index, layer_dim in enumerate(widths[1:] if isFirstInputLayer else widths):
            # construct encoder layer 
            self.encoder_layers.append(Dense(
                    units = layer_dim,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name = name + "_{}".format(layer_index)))
        
    def call(self, inputs):
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x)
        return x


    def build_graph(self, raw_shape):
        x = Input(shape=(raw_shape, ), name = 'encoder_input')
        return tf.keras.Model(inputs = [x], outputs = self.call(x))

"""
Decoder

isTranspose - use transposed layers from encoder. If False create fresh layers of the same size of encoder
"""
class Decoder(tf.keras.layers.Layer):

    """
    Add decoder documentation.
    """
    def __init__(self, encoder, isTranspose : Optional[bool] = True,  name : Optional[str] = "decoder", **kwargs):
        super(Decoder, self).__init__(name = name, **kwargs)
        self.input_dim = 1
        self.output_dim = 1
        self.decoder_layers = []
        if isTranspose:
            for layer_index, layer in enumerate(encoder.layers[1:][::-1]):
                self.decoder_layers.append(
                    DenseTranspose(
                        dense = layer,
                        name = name + "_{}".format(layer_index)
                        )
                    )
        else:
            for layer_index, layer_dim in enumerate([x.input_shape[-1] for x in encoder.layers][1:][::-1]):
                self.decoder_layers.append(Dense(
                    units = layer_dim,
                    kernel_initializer=tf.keras.initializers.GlorotUniform(),
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name = name + "_{}".format(layer_index)))
    """
    Add privileged training: training = None 
    """
    def call(self, inputs):
        x = inputs
        for layer in self.decoder_layers:
            x = layer(x)
        return x


    def build_graph(self, raw_shape):
        x = Input(shape=(raw_shape, ), name = 'decoder_input')
        return tf.keras.Model(inputs = [x], outputs = self.call(x))
    
    """
    Overtide this method to enable serialization 
    """
    def get_config(self):
        pass


"""
Autoencoder
Stack both encoder and decoder
"""
class Autoencoder(tf.keras.Model):

    def __init__(self, widths : List[int] = [32,28,25], name : Optional[str] = "autoencoder", ekwargs : Optional[dict] = {}, dkwargs : Optional[dict] = {}, **kwargs):
        super(Autoencoder, self).__init__(name = name, **kwargs)
        self.input_dim = widths[0]
        self.latent_dim = widths[-1]
        self.encoder = Encoder(widths, **ekwargs).build_graph(widths[0])
        self.decoder = Decoder(self.encoder, **dkwargs).build_graph(widths[-1])
    
    def call(self, input):        
        x = self.encoder.layers[1](input)
        for layer in self.encoder.layers[2:] + self.decoder.layers[1:]:
            x = layer(x)
        return x
        
    
    def compile(self, **kwargs):
        super(Autoencoder, self).compile(**kwargs)


    def build_graph(self,):
        x = Input(shape=(self.input_dim, ), name = 'autoencoder_input')
        return tf.keras.Model(inputs = [x], outputs = self.call(x))