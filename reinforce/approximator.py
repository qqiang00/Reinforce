
#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# function approximators of reinforcment learning

# Author: Qiang Ye
# Date: July 27, 2017

import numpy as np
import tensorflow as tf

from keras import layers, models, optimizers, regularizers, losses
from keras import backend as K

class FuncApproximator(object):
    '''base class of different function approximator subclasses
    '''
    def __init__(self, input_dim = 1, output_dim = 1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        pass

    def __call__(self, x):
        '''return an output given input.
        similar to predict function
        '''
        return self.predict(x)
        #raise NotImplementedError
    
    def predict(self, x):
        '''return an output given input
        '''
        raise NotImplementedError

    def update(self, x, y, **kwargs):
        raise NotImplementedError


class NeuralNetwork(FuncApproximator):
    def __init__(self, input_dim:int = 0,
                       output_dim:int = 0,
                       hidden_dim:int = 16):
        super(NeuralNetwork, self).__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim               # 隐含层特征数
        self._build()

    '''
    def _hidden_layer(self, input):
        net = layers.Dense(units = self.hidden_dim, 
                           use_bias = True)(input)#,
                           #kernel_regularizer = regularizers.l2(self.l2))(input)
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        net = layers.Activation(self.hidden_activation)(net)
        return net
    '''

    def _build(self) -> None:
        states = layers.Input(shape=(self.input_dim,), name = "states")
        net = layers.Dense(units = self.hidden_dim,
                           use_bias=True,
                           activation="relu")(states)
        Q_pred = layers.Dense(units = self.output_dim,
                              use_bias = True)(net)
        
        
        self.model = models.Model(inputs = states, outputs = Q_pred)
        Q_true = layers.Input(shape = (self.output_dim, ))

        self._loss = losses.mean_squared_error(Q_true, Q_pred)


        optimizer = optimizers.Adam(lr=0.1)
        updates_op = optimizer.get_updates(params = self.model.trainable_weights,
                                           constraints = self.model.constraints,
                                           loss = self._loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           Q_true,
                                           K.learning_phase()],
                                   outputs = [self._loss],
                                   updates = updates_op)
        
        

    def update(self, x, y_true, **kwargs):
        x = np.reshape(x,[-1, self.input_dim])
        y_true = np.reshape(y_true, [-1, self.output_dim])
        return self.train_fn(inputs=[x, y_true, 1])


    def predict(self, x):
        x = np.reshape(x, [-1, self.input_dim])
        return self.model.predict(x)

    def copy(self):
        new_net = NeuralNetwork(self.input_dim,
                            self.output_dim,
                            self.hidden_dim)
        #new_net = type(self)()
        #new_net.__dict__.update(self.__dict__)
        weights = self.model.get_weights()
        new_net.model.set_weights(weights)
        return new_net
