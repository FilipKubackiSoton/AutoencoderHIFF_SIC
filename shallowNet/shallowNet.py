import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

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
        
class shallowNet:
    @staticmethod
    def build(input_shape=32, compression=0.8, reg_cof = 0.001, dropout =0.2):
        assert compression <1 and compression >0, "compression coefficient must be between (0,1)" % compression
        assert dropout <1 and dropout >0, "dropout coefficient must be between (0,1)" % dropout
        
        inputs = Input(shape=(input_shape,))
        
        encoder = Dense(
            int(input_shape * compression),
            activation="tanh",
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Zeros(),
            kernel_regularizer=tf.keras.regularizers.l1(reg_cof))

        decoder = DenseTranspose(dense = encoder)
        #the model 
        x = Dropout(dropout)(inputs)
        encoded = encoder(x)
        decoded = decoder(encoded)
        model = tf.keras.Model(inputs, decoded)
        opt = Adam(lr=0.01)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        return model


"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

import numpy as np

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


class Encoder(tf.keras.layers.Layer):

    def __init__(self, input_shape, dropout, reg_cof, compression, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._output_size = int(input_shape * compression)
        self.input_encoder =  Input(shape = (input_shape,1))
        self.drop_encoder = Dropout(dropout)
        self.dense_encoder = Dense(self._output_size,activation="tanh",kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Zeros(),kernel_regularizer=tf.keras.regularizers.l1(reg_cof))
    
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x = self.drop_encoder(inputs) 
        x = self.dense_encoder(x)
        return x

    def get_dense_encoder(self):
        return self.dense_encoder

    @property
    def output_size(self):
        return self._output_size 

class Decoder(tf.keras.layers.Layer):

    def __init__(self, encoder, **kwargs ):
        super(Decoder, self).__init__(**kwargs)
        self.encoder_d = encoder.get_dense_encoder() 

    def build(self, input_shape):
        self.dense_decoder = DenseTranspose(dense = self.encoder_d)

    def call(self, inputs):
        x = self.dense_decoder(inputs)
        return x

class Autoencoder(tf.keras.Model):

    def __init__(self, input_size=32, compression = 0.8, dropout = 0.2, reg_cof = 0.001, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.input_size = input_size
        self.compression = 0.8
        self.dropout = dropout
        self.reg_cof = reg_cof 
        self.encoder = Encoder(
            input_shape = self.input_size,
            dropout = self.dropout, 
            reg_cof = self.reg_cof,
            compression = self.compression, 
            )

    def build(self, _):
        self.encoder = Encoder(
            input_shape = self.input_size,
            dropout = self.dropout, 
            reg_cof = self.reg_cof,
            compression = self.compression, 
            )
        self.decoder = Decoder(
            encoder = self.encoder
            ) 

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x

    def encode(self, X):
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

     
class shallowNet:

    @staticmethod
    def build(input_shape=32, compression=0.8, reg_cof = 0.01, dropout =0.2):
        assert compression <1 and compression >0, "compression coefficient must be between (0,1)" % compression
        assert dropout <1 and dropout >0, "dropout coefficient must be between (0,1)" % dropout
        
        inputs = Input(shape=(input_shape,))
        encoder = Dense(int(input_shape * compression),activation="tanh",kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Zeros(),kernel_regularizer=tf.keras.regularizers.l1(reg_cof))
        decoder = DenseTranspose(dense = encoder)

        #the model 
        x = Dropout(dropout)(inputs)
        encoded = encoder(x)
        decoded = decoder(encoded)
        model = tf.keras.Model(inputs, decoded)
        opt = Adam(lr=0.01)
        model.compile(loss='mse', optimizer=opt)
        model.summary()

        return model

"""