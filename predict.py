# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:03:54 2017

@author: Alexandre Boyker
"""

import os 
from generator import BatchGenerator
from cnn import CnnTextClassifier
from parser2 import *
import matplotlib.pyplot as plt
from utilities import plot_confusion_matrix
from argparse import ArgumentParser




parser = ArgumentParser()

parser.add_argument("-bt", "--batch_size_predict", dest="batch_size_predict",
                    help="batch size for prediction, default=128", type=int, default=800)


parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                    help="learning rate,  default=.001", type=float, default=.001)


parser.add_argument("-e", "--embedding_size",
                     dest="embedding_size", default=128, type=int,
                    help="the dimensions of the embedding for the input, default=128")

parser.add_argument("-d", "--dropout_proba",
                     dest="dropout_proba", default=.0, type=float,
                    help="drop out probability, default=.0 for prediction")


parser.add_argument("-n", "--num_classes",
                     dest="num_classes", default=16, type=int,
                    help="number of classes to predict, default=16")

parser.add_argument("-ev", "--evaluation_every",
                     dest="evaluation_every", default=100, type=int,
                    help="number of training steps required to perform new evaluation, default=100")

parser.add_argument("-ep", "--num_epochs",
                     dest="num_epochs", default=1, type=int,
                    help="number of epochs, default=100")

parser.add_argument("-nv", "--build_voc",
                     dest="build_voc", default=False,
                    help="build new vocabulary mapping, default=False")

parser.add_argument("-v", "--voc",
                     dest="voc_path", default='voc',
                    help="path of vocabulary files, default='voc'")

parser.add_argument("-ms", "--max_sentence",
                    dest="max_sentence_length", default=100,type=int,
                    help="maximum size of the input sentences, longer sentences are truncated, default=100")

args = parser.parse_args()

batch_size_predict = args.batch_size_predict
learning_rate = args.learning_rate
embedding_size = args.embedding_size
dropout_proba = args.dropout_proba
num_classes = args.num_classes
evaluation_every = args.evaluation_every
num_epochs = args.num_epochs
voc_path = args.voc_path
batch_size_predict = args.batch_size_predict
build_voc = args.build_voc
max_sentence_length = args.max_sentence_length

cnn_hyperparameters = {'learning_rate':learning_rate, 'embedding_size':embedding_size,
                  'dropout_proba':dropout_proba, 'num_classes':num_classes,
                 'evaluation_every':evaluation_every, 'num_epochs':num_epochs}

def get_data_paths(train_or_test):
    data_path = os.path.join(os.getcwd(), 'data', train_or_test)
    file_list =[]
    for file in os.listdir(data_path):
        if file.endswith(".txt"): 
            file_list.append( os.path.join(data_path ,file))
    return file_list

def predict():
    
    file_list_predict = get_data_paths('predict')
    line_parser = MbtiParser()
    predict_generator = BatchGenerator(file_list_predict, line_parser, 
                                    batch_size=batch_size_predict, build_voc = False, max_sentence_length=max_sentence_length,
                                    voc_path = voc_path)
     

    cnn = CnnTextClassifier(**cnn_hyperparameters)
    
    pred_dic = cnn.predict(predict_generator)
    
    plt.figure()
    
    plot_confusion_matrix(pred_dic['ground_truth'],pred_dic['predictions'] , list(range(num_classes)),
                      title='Confusion matrix, without normalization')

    plt.figure()
    
    plot_confusion_matrix(pred_dic['ground_truth'],pred_dic['predictions'] , list(range(num_classes)),
                      title='Confusion matrix, with normalization', normalize=True)
    plt.show()

if __name__ == '__main__':
    
    predict()
    
    