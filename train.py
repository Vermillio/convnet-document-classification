# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:21:27 2017

@author: Alexandre Boyker
"""
import os 
from generator import BatchGenerator
from cnn import CnnTextClassifier
from parser2 import *
from argparse import ArgumentParser



""" -------------------Parameters definition------------------- """

parser = ArgumentParser()

parser.add_argument("-bt", "--batch_size_train", dest="batch_size_train",
                    help="batch size for training, default=128", type=int, default=128)

parser.add_argument("-bv", "--batch_size_test", dest="batch_size_test",
                    help="batch size for validation data, default=128", type=int, default=128)

parser.add_argument("-lr", "--learning_rate", dest="learning_rate",
                    help="learning rate,  default=.001", type=float, default=.001)


parser.add_argument("-e", "--embedding_size",
                     dest="embedding_size", default=128, type=int,
                    help="the dimensions of the embedding for the input, default=128")

parser.add_argument("-d", "--dropout_proba",
                     dest="dropout_proba", default=.25, type=float,
                    help="drop out probability, default=.25")


parser.add_argument("-n", "--num_classes",
                     dest="num_classes", default=16, type=int,
                    help="number of classes to predict, default=16")

parser.add_argument("-ev", "--evaluation_every",
                     dest="evaluation_every", default=100, type=int,
                    help="number of training steps required to perform new evaluation, default=100")

parser.add_argument("-ep", "--num_epochs",
                     dest="num_epochs", default=100, type=int,
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

batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
learning_rate = args.learning_rate
embedding_size = args.embedding_size
dropout_proba = args.dropout_proba
num_classes = args.num_classes
evaluation_every = args.evaluation_every
num_epochs = args.num_epochs
voc_path = args.voc_path
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
build_voc = args.build_voc
max_sentence_length = args.max_sentence_length

def get_data_paths(train_or_test):
    data_path = os.path.join(os.getcwd(), 'data', train_or_test)
    file_list =[]
    for file in os.listdir(data_path):
        if file.endswith(".txt"): 
            file_list.append( os.path.join(data_path ,file))
    return file_list




cnn_hyperparameters = {'learning_rate':learning_rate, 'embedding_size':embedding_size,
                  'dropout_proba':dropout_proba, 'num_classes':num_classes,
                 'evaluation_every':evaluation_every, 'num_epochs':num_epochs}



""" -------------------Train function------------------- """


def train():

    voc_path = 'voc'
    
    file_list_train = get_data_paths('train')
    file_list_test = get_data_paths('test')

    line_parser = MbtiParser()
        
    train_generator = BatchGenerator(file_list_train, line_parser, batch_size=batch_size_train, 
                                     build_voc = build_voc, voc_path = voc_path, max_sentence_length=max_sentence_length)
                                     
     
     #Do not build vocabulary for testing
    test_generator = BatchGenerator(file_list_test, line_parser, 
                                    batch_size=batch_size_test, build_voc = False,max_sentence_length=max_sentence_length,
                                    voc_path = voc_path)
    

    print("vocab_size {}".format(train_generator.vocab_size))
    
    cnn = CnnTextClassifier(**cnn_hyperparameters)
    cnn.fit_generator(train_generator, test_generator)

if __name__ == '__main__':
    
    train()
