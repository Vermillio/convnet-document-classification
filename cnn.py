# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 17:45:33 2017

@author: Alexandre Boyker
"""

from datetime import datetime
import tensorflow as tf
import os
import numpy as np
from utilities import *
import matplotlib.pyplot as plt

class CnnTextClassifier(object):
    
    
    def __init__(self, learning_rate=.001, embedding_size =128,
                  dropout_proba=.25, num_classes=2,
                  evaluation_every=100, num_epochs=5, num_features_maps=64, 
                  filter_heigth_list=[3,4,5], vocab_size = 1000):
        
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.dropout_proba = dropout_proba
        self.num_classes = num_classes
        self.evaluation_every = evaluation_every
        self.num_epochs = num_epochs
        self.num_features_maps = num_features_maps
        self.filter_heigth_list = filter_heigth_list
        self.vocab_size = vocab_size
        self.sequence_length = 50
        
    def init_weights(self,shape):
        
        init_random_dist = tf.truncated_normal(shape, stddev=0.1)
        
        return tf.Variable(init_random_dist)


    def init_bias(self,shape):
        
        init_bias_vals = tf.constant(0.1, shape=shape)
        
        return tf.Variable(init_bias_vals)
    
    
    def conv2d(self,x, W):
        
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    
    def max_pool_2by2(self, x, filter_heigth):
        
        return tf.nn.max_pool(x, ksize=[1, self.sequence_length - filter_heigth + 1, 1, 1],
                              strides=[1, 1, 1, 1], padding='VALID')
    
    def convolutional_layer(self,input_x, shape):
        
        W = self.init_weights(shape) 
        b = self.init_bias([shape[3]])
        
        return tf.nn.relu(self.conv2d(input_x, W)+b)
    
    def normal_full_layer(self,input_layer, size, layer_name="normal_layer"):
        
        with tf.name_scope(layer_name):
    
        
            input_size = int(input_layer.get_shape()[1])
            
            with tf.name_scope('weights'):
            
                W = self.init_weights([input_size, size])
            
            with tf.name_scope('biases'):
    
                b = self.init_bias([size])
    
        return tf.matmul(input_layer, W) + b
    
    
    def build_model(self):
        
        
        # Placeholders for input, output and dropout
        input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
    
                W = tf.Variable(
                        tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                        name="W")
                embedded_chars = tf.nn.embedding_lookup(W, input_x)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        
        layers_to_concatenate = []
        
        for filter_heigth in self.filter_heigth_list:
            
            convo = self.convolutional_layer(embedded_chars_expanded, shape=[filter_heigth, self.embedding_size,1,self.num_features_maps])
            convo_pooling = self.max_pool_2by2(convo,filter_heigth)
            flat = tf.contrib.layers.flatten(convo_pooling)
            print("flattened array shape",flat.get_shape())
            layers_to_concatenate.append(flat)
       
        
        combined_branch = tf.concat(layers_to_concatenate, axis=1)
        
        
        h_drop = tf.nn.dropout(combined_branch, self.dropout_proba )
        y_pred = self.normal_full_layer(h_drop, self.num_classes, layer_name='y_pred')
        
        
        with tf.name_scope('cross_entropy'):
            
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=input_y,logits=y_pred))
        
        tf.summary.scalar('cross_entropy', cross_entropy)
        

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = optimizer.minimize(cross_entropy)
        
        predi = tf.argmax(y_pred,1, name ='predictions')
        
        with tf.name_scope('accuracy_1'):
                       
            with tf.name_scope('correct_prediction'):
                 
                
                matches = tf.equal(predi,tf.argmax(input_y,1))
            
            with tf.name_scope('accuracy'):
        
                acc = tf.reduce_mean(tf.cast(matches,tf.float32))
                
        tf.summary.scalar('accuracy_1', acc)

        return train, cross_entropy, acc, input_x, input_y, predi
        
    
    def fit_generator(self, generator_train, generator_test= None):
        
        try:
            
            self.sequence_length = generator_train.sequence_len          
            self.vocab_size = generator_train.vocab_size           
            print("vocab_size {}".format( self.vocab_size))
            print("num classes {}".format( self.num_classes))
           
        except Exception as e:
            
            print(str(e))
        
                
        
        train, cross_entropy,acc, input_x, input_y, y_pred = self.build_model()
        init = tf.global_variables_initializer()
        steps = generator_train.len_doc//generator_train.batch_size
        
        with tf.Session() as sess:
            
            
            saver = tf.train.Saver()
            sess.run(init)

            for i, batch in enumerate(generator_train.__iter__()):
                
                x_batch_train, y_batch_train = batch
                
                tr, ce, ac =sess.run([train,cross_entropy,acc],feed_dict={input_x:x_batch_train, input_y:y_batch_train})
                print("{} iterations: {} out of {}  loss: {}  accuracy: {}".format(str(datetime.now()),1+i,self.num_epochs*steps, ce, ac))
                
                validation_steps = generator_test.len_doc//generator_test.batch_size
                
                if (i % self.evaluation_every == 0) and (i!=0):
                    predi_list = []
                    ground_truth_list = []
                    print("\nEvaluating model...:")
                    for ind,valid_batch in enumerate(generator_test.__iter__()):
                        x_batch_valid, y_batch_valid = valid_batch
                        number_y= 0
                        cnt_y = 0
                        for item in y_batch_valid:
                            number_y += item[0]
                            cnt_y += 1
                        print("proportion of one", number_y/y_batch_valid.shape[0])
                        print("number of validation samples", cnt_y)
                        tr, ce, ac =sess.run([train, cross_entropy, acc], feed_dict={input_x:x_batch_valid, input_y:y_batch_valid})
                        print(x_batch_valid.shape)
                        print( "{} iterations: {} out of {}  validation loss: {} validation accuracy {}".format( str(datetime.now()),1+ind,validation_steps , ce, ac ))
                        print("")
                        predi =sess.run(y_pred, feed_dict={input_x:x_batch_valid})
                        predi_list += list(predi)
                        ground_truth_list +=  [np.argmax(item) for item in y_batch_valid]
                        
                        if ind  >= validation_steps - 1: 
                            
                            break
    
                    print('\n')
                if i == self.num_epochs*steps:
                    
                    if not os.path.exists(os.path.join(os.getcwd(), 'saved_model')):
                        os.makedirs(os.path.join(os.getcwd(), 'saved_model'))
                    saver.save(sess, os.path.join(os.getcwd(), 'saved_model','my_test_model'))
                    #predi =sess.run(y_pred, feed_dict={input_x:x_batch_valid})
                    plt.figure()
                    plot_confusion_matrix(ground_truth_list, predi_list, range(self.num_classes)                          ,normalize=False,   title='Confusion matrix')
                    plt.show()
                    break
                
    def predict(self, generator):
        
        sess=tf.Session()   
        saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), 'saved_model','my_test_model.meta'))
        saver.restore(sess,tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'saved_model')))
        
        
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("input_x:0")
        y_pred = graph.get_operation_by_name("predictions").outputs[0]
        
        
        validation_steps = generator.len_doc//generator.batch_size
                
        print("\nMaking predictions...:")
        
        predi_list = []
        ground_truth_list = []
        
        for ind,valid_batch in enumerate(generator.__iter__()):
            x_batch_valid, y_batch_valid = valid_batch
            number_y= 0
            cnt_y = 0
            for item in y_batch_valid:
                number_y += item[0]
                cnt_y +=1
                
            print("proportion of one",number_y/y_batch_valid.shape[0])
            print("number of validation samples",cnt_y)
            print("running sess")
            predi =sess.run(y_pred, feed_dict={input_x:x_batch_valid})
            predi_list += list(predi)
            ground_truth_list +=  [np.argmax(item) for item in y_batch_valid]
            if ind  == validation_steps - 1: break
        
        pred_dic ={}
        ground_truth = [np.argmax(item) for item in y_batch_valid]
        pred_dic = {'predictions':list(predi), 'ground_truth':ground_truth }
        plt.figure()
        plot_confusion_matrix(ground_truth, predi, range(self.num_classes),
                                              normalize=False,
                                              title='Confusion matrix')
        plt.show()
        return pred_dic
