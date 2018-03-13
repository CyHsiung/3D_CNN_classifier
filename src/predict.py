import numpy as np
import tensorflow as tf
from os.path import join
import os
from load_data import test_load_test

project_dir = os.path.abspath('./')
while project_dir[-3:] != 'src':
    project_dir = os.path.abspath(join(project_dir, os.pardir))
project_dir = join(project_dir, '..')
corpus_dir = join(project_dir, 'corpus')
models_dir = join(project_dir, 'models')

loadModel_dir = '3dcnn_180313_132858'
loadModelName = 'acc_0.90909'

loadModel_dir = join(models_dir, loadModel_dir)


def predict(x_test, y_test, batch_size = 1):
    model_saver = tf.train.import_meta_graph(join(loadModel_dir, loadModelName + '.meta'))
    with tf.Session() as sess:
        acc = 0
        numValBatches = int(len(x_test)/batch_size)
        model_saver.restore(sess, join(loadModel_dir, loadModelName))
        graph = tf.get_default_graph()
        x_input = graph.get_tensor_by_name("input:0")
        y_input = graph.get_tensor_by_name("output:0")
        accuracy = graph.get_tensor_by_name('accuracy:0')
        # test data accuracy
        for itr in range(numValBatches):
            mini_batch_x_test = x_test[itr * batch_size: min((itr + 1) * batch_size, len(x_test))]
            mini_batch_y_test = y_test[itr * batch_size: min((itr + 1) * batch_size, len(y_test))]
            if not mini_batch_x_test:
                continue
            acc += sess.run(accuracy, feed_dict={x_input: mini_batch_x_test, y_input: mini_batch_y_test})
        valAcc = round(acc / numValBatches, 5)
        print(valAcc)
        
        #predicted label
        predicted_label = graph.get_tensor_by_name('predicted_label:0')
        print(sess.run(predicted_label, feed_dict={x_input: x_test[5:10], y_input: y_test[5:10]}))

if __name__ == '__main__':
    x_test, y_test = test_load_test()
    predict(x_test, y_test)
    
