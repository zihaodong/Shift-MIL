
from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras import initializers
import tensorflow as tf


class Shift_MILLayer(Layer):

    def __init__(self, weights=None, b_init = 'zero', a_init = 'one', **kwargs):

        self.a_init = initializers.Ones()([1])
        self.b_init = initializers.Zeros()([1,2])
        self.initial_weights = weights
        
        super(Shift_MILLayer, self).__init__(**kwargs)


    def build(self, input_shape):

        self.input_spec = [InputSpec(shape=input_shape)]

        # Compatibility with TensorFlow >= 1.0.0
        self.a_1 = K.variable(self.a_init, name='a')
        self.b_1 = K.variable(self.b_init, name='b')

        self.trainable_weights = [self.a_1, self.b_1]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, inputs):
        
        # Shift MIL operation
        mean = K.mean(inputs, axis=[1])
        ss = 0.97
        output = (K.sigmoid(self.a_1 * (mean ** ss - self.b_1)) - K.sigmoid(-self.a_1 * self.b_1)) / (
            K.sigmoid(self.a_1 * (1 - self.b_1)) - K.sigmoid(-self.a_1 * self.b_1))
        #print(output)
        bsize, a = output.get_shape().as_list()
        bsize = K.shape(output)[0]
        outtf = K.reshape(output, [bsize,1,a]) 
        #print(outtf.shape)
        return outtf

    def compute_output_shape(self, input_shape):
        input_shape = self.input_spec[0].shape
        return input_shape


