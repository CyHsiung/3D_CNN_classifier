import numpy as np
import tensorflow as tf
from model import cnn_model
from load_data import test_load_data
import os
from os.path import join

import datetime


n_class = 2

project_dir = os.path.abspath('./')
while project_dir[-3:] != 'src':
    project_dir = os.path.abspath(join(project_dir, os.pardir))
project_dir = join(project_dir, '..')
corpus_dir = join(project_dir, 'corpus')
models_dir = join(project_dir, 'models')
saveModel_dir = join(models_dir, '3dcnn_' + datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
os.makedirs(saveModel_dir)

def train_neural_network(x_train, y_train, x_val, y_val, learning_rate = 0.05, drop_rate = 0.7, epochs = 10, batch_size = 1):
    x_input = tf.placeholder(tf.float32, shape=[None, None, None, None, 1], name = 'input')
    y_input = tf.placeholder(tf.float32, shape=[None, n_class], name = 'output')
    drop_prob = tf.placeholder(tf.float32, shape = None)
    with tf.name_scope("cross_entropy"):
        prediction = cnn_model(x_input, drop_prob, seed = 42)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_input))
                              
    with tf.name_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    predicted_label = tf.argmax(prediction, 1, name = 'predicted_label')    
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_input, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'), name = 'accuracy')
    
    iterations = int(len(x_train)/batch_size)
    
    # to save model
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_time = datetime.datetime.now()

        iterations = int(len(x_train)/batch_size) + 1
        # run epochs

        maxAcc = 0
        for epoch in range(epochs):
            start_time_epoch = datetime.datetime.now()
            print('Epoch: ', epoch)
            epoch_loss = 0
            # mini batch
            for itr in range(iterations):
                mini_batch_x = x_train[itr * batch_size: min((itr + 1)*batch_size, len(x_train))]
                mini_batch_y = y_train[itr * batch_size: min((itr + 1)*batch_size, len(y_train))]
                if not mini_batch_x:
                    continue
                _optimizer, _cost = sess.run([optimizer, cost], feed_dict={x_input: mini_batch_x, y_input: mini_batch_y, drop_prob: drop_rate})
                epoch_loss += _cost

            #  using mini batch in case not enough memory
            acc = 0
            numValBatches = int(len(x_val)/batch_size) + 1
            for itr in range(numValBatches):
                mini_batch_x_val = x_val[itr * batch_size: min((itr + 1) * batch_size, len(x_val))]
                mini_batch_y_val = y_val[itr * batch_size: min((itr + 1) * batch_size, len(y_val))]
                if not mini_batch_x_val:
                    continue
                acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_val, y_input: mini_batch_y_val})
            valAcc = round(acc / numValBatches, 5)
            end_time_epoch = datetime.datetime.now()
            print(' Testing Set Accuracy:', valAcc, ' Time elapse: ', str(end_time_epoch - start_time_epoch))
            if valAcc > maxAcc:
                # save model when better performance
                saver.save(sess, join(saveModel_dir, 'acc_' + str(valAcc)))
                maxAcc = valAcc 

        end_time = datetime.datetime.now()
        print('Time elapse: ', str(end_time - start_time))

if __name__ == '__main__':
    x_train, y_train, x_val, y_val = test_load_data()
    train_neural_network(x_train, y_train, x_val, y_val)
    
    
    
