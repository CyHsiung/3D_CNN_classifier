import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

def cnn_model(x_train, drop_rate=0.7, seed=None):
    
    with tf.name_scope("layer_conv"):
        conv_1 = tf.layers.conv3d(inputs = x_train, filters = 16, kernel_size = [8, 8, 8], strides = [2, 2, 2], padding = 'same', activation = tf.nn.relu)
        conv_bn_1 = tf.layers.batch_normalization(inputs = conv_1, training=True)
        
        conv_2 = tf.layers.conv3d(inputs = conv_bn_1, filters = 32, kernel_size = [6, 6, 6], strides = [2, 2, 2], padding = 'same', activation = tf.nn.relu)
        conv_bn_2 = tf.layers.batch_normalization(inputs = conv_2, training=True)
        
        conv_3 = tf.layers.conv3d(inputs = conv_bn_2, filters = 64, kernel_size = [4, 4, 4], strides = [2, 2, 2], padding = 'same', activation = tf.nn.relu)
        conv_bn_3 = tf.layers.batch_normalization(inputs = conv_3, training=True)
        
        conv_4 = tf.layers.conv3d(inputs = conv_bn_3, filters = 128, kernel_size = [3, 3, 3], strides = [2, 2, 2], padding = 'same', activation = tf.nn.relu)
        conv_bn_4 = tf.layers.batch_normalization(inputs = conv_4, training=True)
        
    with tf.name_scope("fully_connected"):
        global_max_pool = tf.reduce_max(input_tensor = conv_bn_4, axis = [1, 2, 3])
        flatten = tf.contrib.layers.flatten(inputs = global_max_pool)
        # flatten = tf.contrib.layers.flatten(inputs = cnn3s_bn)
        dense_1 = tf.layers.dense(inputs = flatten, units = 64, activation = tf.nn.relu)
        # (1-keep_rate) is the probability that the node will be kept
        dropout = tf.layers.dropout(inputs = dense_1, rate = drop_rate, training=True)
        
    with tf.name_scope("y_conv"):
        y_conv = tf.layers.dense(inputs = dropout, units = 2)
    
    return y_conv

    
