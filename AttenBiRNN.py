import pandas as pd
import numpy as np
import glob
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import warnings
warnings.filterwarnings('ignore')



import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa


class MyAttention(keras.layers.Layer):
    def __init__(self,
                 attention_size, 
                 **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.attention_size = attention_size

    def build(self, input_shape):
        initializer = tf.random_normal_initializer(stddev=0.1)
        self.w_omega = tf.Variable(name='atten_kernel1', initial_value=initializer(shape=(input_shape[-1], self.attention_size)), dtype='float32', trainable=True)
        self.b_omega = tf.Variable(name='atten_bias', initial_value=initializer(shape=(self.attention_size,)), dtype='float32', trainable=True)
        self.u_omega = tf.Variable(name='atten_kernel2', initial_value=initializer(shape=(self.attention_size,)), dtype='float32', trainable=True)

    def call(self, inputs, time_major=False, return_alphas=False):
        if isinstance(inputs, tuple):
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T, B, D) => (B, T, D)
            inputs = tf.transpose(inputs, [1, 0, 2])
            
        with tf.name_scope('v'):
            # applying fully connected layer with non-linear activation to each of the B*T timestamps
            # the shape of v is (B, T, D)*(D, A) = (B, T, A)
            v = tf.tanh(tf.tensordot(inputs, self.w_omega, axes=1) + self.b_omega)
            
        # for each of the timesteps its vector of size A from 'v' is reduced with 'u' vector
        # the shape of vu is (B,T,A)*(A) = (B, T)
        vu = tf.tensordot(v, self.u_omega, axes=1, name='vu')
        alphas = tf.nn.softmax(vu, name='alphas')

        # output: (B, T, D)*(B, T, *) = (B, T, D) -> sum over T => (B, D)
        output = tf.reduce_sum(inputs*tf.expand_dims(alphas, -1), 1)
            
        if not return_alphas:
            return output
        return output, alphas
    
    
    
class AttenBiRNN:
    def __init__(self,
                 is_train=False,
                 batch_size=1,
                 n_1d_layer=20,
                 n_2d_layer=12,
                 dilation=[1,2,4,8],
                 n_feat_1d=64,
                 n_bottle_1d=32,
                 n_feat_2d=128,
                 n_bottle_2d=64,
                 n_hidden_rnn=64,
                 attention_size = 50,
                 kernel_size=3,
                 p_dropout=0.2,
                 l2_coef=0.0005,
                 use_fragment_profile=True,
                 train_pssm=False):
        self.is_train = is_train
        self.batch_size = batch_size
        self.n_1d_layer = n_1d_layer
        self.n_2d_layer = n_2d_layer
        self.dilation = dilation
        self.n_feat_1d = n_feat_1d
        self.n_bottle_1d = n_bottle_1d
        self.n_feat_2d = n_feat_2d
        self.n_bottle_2d = n_bottle_2d
        self.kernel_size = kernel_size
        self.n_hidden_rnn = n_hidden_rnn
        self.attention_size = attention_size
        self.p_dropout = p_dropout
        self.l2_coef = l2_coef
        self.use_fragment_profile = use_fragment_profile
        self.train_pssm = train_pssm
        
        self.build_model()
        
    def build_model( self ):
        
        #inst_norm = tfa.layers.InstanceNormalization()
        #inst_norm = keras.layers.BatchNormalization(axis=[0,-1])
        
        if self.use_fragment_profile:
            self.feat_1d = keras.Input(shape=(None, 20+11), batch_size=self.batch_size, name='feat_1d', dtype=tf.float32)
            self.feat_2d = keras.Input(shape=(None, None, 9), batch_size=self.batch_size, name='feat_2d', dtype=tf.float32)
            n1D_dimen = 20+11
            n2D_dimen = 9
        else:
            self.feat_1d = keras.Input(shape=(None, 11), batch_size=self.batch_size, name='feat_1d', dtype=tf.float32)
            self.feat_2d = keras.Input(shape=(None, None, 9), batch_size=self.batch_size, name='feat_2d', dtype=tf.float32)
            n1D_dimen = 11
            n2D_dimen = 9
                
        self.n_res = tf.shape(self.feat_1d)[1]
        
        # projection to n_feat_1d
        feat = keras.layers.Conv1D(self.n_feat_1d, 1, padding='same')(self.feat_1d)
        
        #============================	
        # 1D ResNet with combined features
        #============================
        for i in range(self.n_1d_layer):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block_1d(feat, step=i, dilation=d)
        feat = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(feat, training=True))
        
        # one body
        one_body = tf.tile(feat, [1,self.n_res,1])
        one_body = tf.reshape(one_body, [self.batch_size, self.n_res, self.n_res, self.n_feat_1d])
        
        # two body
        two_body = keras.layers.Conv2D(self.n_feat_2d, 1, padding='same')(self.feat_2d)
        two_body = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(two_body, training=True))
        
        #
        feat = tf.concat((two_body, one_body, tf.transpose(one_body, (0,2,1,3))), axis=-1)
        
        #
        # Add 2D resnet here!!!
        feat = keras.layers.Conv2D( self.n_feat_2d, 1, padding="same", use_bias=False)(feat)
        #=================================
        # 2D ResNet with combined features
        #=================================
        # Stacking 2-dim residual blocks (receptive field size: 61) 
        for i in range(self.n_2d_layer):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block_2d(feat, step=i, dilation=d)
        feat = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(feat, training=True))
        
        #
        #========================================
        # LSTM to extract 1-dimentional features from input features
        #========================================
        feat, self.alphas = self.BiLSTM_w_attention(feat)
        feat = keras.layers.Dense(self.n_feat_1d, activation=None, use_bias=True)(feat)
        feat = tf.reshape( feat, [self.batch_size,self.n_res, self.n_feat_1d])
            
        feat = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(feat, training=True))
        
        # Stacking 1-dim residual blocks
        for i in range(4):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block_1d(feat, step=i, dilation=d)
        feat = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(feat, training=True))

        # final cov
        logits = keras.layers.Conv1D(20, 1, padding='same')(feat) # to 20 amino acids
        
        # define the model
        self.model = keras.Model(inputs=[self.feat_1d, self.feat_2d], outputs=logits, name='AttenBiRNN')
        
        ## doesn't work very well
        ## keras.utils.plot_model(self.model,'./model.png', show_shapes=True)
    
    def ResNet_block_1d( self, x, step=0, dilation=1 ):
        
        shortcut = x
        # bottleneck layer (kernel: 1, n_feat_1d => n_bottle_1d)
        x = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(x, training=True))
        x = keras.layers.Conv1D(self.n_bottle_1d, 1, padding='same')(x)
        x = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(x, training=True))
        # convolution
        x = keras.layers.Conv1D(self.n_bottle_1d, self.kernel_size, dilation_rate=dilation, padding='same')(x)
        x = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(x, training=True))
        x = keras.layers.Dropout(rate=self.p_dropout)(x, training=self.is_train)
        x = keras.layers.Conv1D(self.n_feat_1d, 1, padding='same')(x)
        x += shortcut
        
        return x
    
    def ResNet_block_2d(self, x, step=0, dilation=1): # bottleneck block w/ pre-activation
        
        shortcut = x
        # bottleneck layer (kernel: 1, n_feat_2d => n_bottle_2d)
        x = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(x, training=True))
        x = keras.layers.Conv2D(self.n_bottle_2d, 1, padding='same')(x)
        x = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(x, training=True))
        # convolution
        x = keras.layers.Conv2D(self.n_bottle_2d, self.kernel_size, dilation_rate=dilation, padding='same')(x)
        x = tf.nn.elu(keras.layers.BatchNormalization(axis=[0,-1])(x, training=True))
        x = keras.layers.Dropout(rate=self.p_dropout)(x, training=self.is_train)
        # project up (kernel: 1, n_bottle_2d => n_feat_2d)
        x = keras.layers.Conv2D(self.n_feat_2d, 1, padding='same')(x)
        x += shortcut
        
        return x
    
    def BiLSTM_w_attention(self, x, return_alphas=True):

        # tensorflow2 [batch, timesteps, feature]
        x = tf.reshape(x, [self.batch_size*self.n_res, self.n_res, self.n_feat_2d])
        
        # define a lstm cell
        #The importable implementations have been deprecated - instead, LSTM and GRU will default to CuDNNLSTM and CuDNNGRU if all conditions are met:
        # The requirements to use the cuDNN implementation are:
        #1. `activation` == `tanh`
        #2. `recurrent_activation` == `sigmoid`
        #3. `recurrent_dropout` == 0
        #4. `unroll` is `False`
        #5. `use_bias` is `True`
        #6. Inputs are not masked or strictly right padded.
        #7. reset_after = True (GRU only)
        
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(self.n_hidden_rnn, 
                                                            activation='tanh', 
                                                            recurrent_activation='sigmoid', 
                                                            recurrent_dropout=0, 
                                                            unroll=False, 
                                                            use_bias=True,
                                                            return_sequences=True), merge_mode='concat')

        # get BiLSTM cell output
        outputs = lstm(x)

        # apply attention
        outputs, alphas = MyAttention(self.attention_size)(outputs, time_major=False, return_alphas=return_alphas)
        
        return outputs, alphas
    
    def load( self, model_fn ):
        if os.path.exists(f"{model_fn}.h5"):
            self.model.load_weights(f"{model_fn}.h5")
            return True
        return False
    
    # return the probabilitis of each amino acid
    def __call__(self, feat1d, feat2d):
        logits = self.model([feat1d, feat2d])
        results = tf.nn.softmax(logits, axis=-1)

        return results
