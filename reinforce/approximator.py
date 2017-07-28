
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

    def fit(self, x, y, **kwargs):
        raise NotImplementedError

class LinearFunc(FuncApproximator):
    def __init__(self, input_dim=1, output_dim = 1):
        super(LinearFunc,self).__init__(input_dim, output_dim)


class NeuralNetwork(FuncApproximator):
    def __init__(self, input_dim:int = 0,
                       output_dim:int = 0,
                       hidden_dim:int = 10,
                       hidden_layers:int = 1,
                       output_bound = None,
                       hidden_activation = "relu",
                       output_activation = None,#"tanh",
                       use_batchnorm:bool = True
                       ):
        super(NeuralNetwork, self).__init__(input_dim,output_dim)
        # self.input_dim = input_dim              # 输入层特征数 feature num of input
        # self.output_dim = output_dim             # 输出层特征数
        self.hidden_dim = hidden_dim               # 隐含层特征数
        self.hidden_layers = hidden_layers if hidden_layers > 0 else 1
        # self.output_bound = output_bound  # 输出值的边界
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batchnorm = use_batchnorm  # 是否使用批归一化
        
        self.l2 = 0.01
        self.clipvalue = 40
        self._build()

    def _hidden_layer(self, input):
        net = layers.Dense(units = self.hidden_dim,
                           kernel_regularizer = regularizers.l2(self.l2))(input)
        if self.use_batchnorm:
            net = layers.BatchNormalization()(net)
        net = layers.Activation(self.hidden_activation)(net)
        return net

    def _build(self) -> None:
        states = layers.Input(shape=(self.input_dim,), name = "states")
        net = self._hidden_layer(states) # at least one hidden layer

        # build hidden_layer
        for i in range(self.hidden_layers - 1):
            net = self._hidden_layer(net)

        # output layer: linear activation
        Q_pred = layers.Dense(units = self.output_dim,
                           kernel_regularizer = regularizers.l2(self.l2))(net)
        
        if self.output_activation is not None:
            Q_pred = layers.Activation(self.output_activation)(Q_pred)
        if self.output_bound is not None:
            Q_pred = layers.Lambda(lambda x: x * self.output_bound)(Q_pred)
        
        self.model = models.Model(inputs = states, outputs = Q_pred)
        Q_true = layers.Input(shape = (self.output_dim, ))

        loss = losses.mean_square_error(Q_true, Q_pred)
        for l2_reg_loss in self.model.losses:
            loss += l2_reg_loss

        optimizer = optimizers.Adam(self.clipvalue)
        updates_op = optimizer.get_updates(params = self.model.trainable_weights,
                                           constraints = self.model.constraints,
                                           loss = loss)

        self.train_fn = K.function(inputs=[self.model.inputs,
                                           Q_true,
                                           K.learning_phase()],
                                   outputs = [],
                                   updates = updates_op)
        

    def fit(self, x, y_label, **kwargs):
        self.train_fn(inputs=[x, y_label, 1])


    def __copy__(self):
        #net = NeuralNetwork(self.input_dim,
        #                    self.output_dim,
        #                    self.hidden_dim,
        #                    self.hidden_layers,
        #                    self.output_bound,
        #                    self.hidden_activation,
        #                    self.output_activation,
        #                    self.use_batchnorm)
        new_net = type(self)()
        new_net.__dict__.update(self.__dict__)
        weights = self.model.get_weights()
        new_net.model.set_weights(weights)
        return net
