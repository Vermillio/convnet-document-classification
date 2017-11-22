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

# Add more options if you like
parser.add_argument("-b", "--batch_size", dest="batch_size_predict",
                    help="batch size for predictions", type=int)
parser.add_argument("-v", "--voc",
                     dest="voc_path", default='voc',
                    help="path of vocabulary files")

args = parser.parse_args()



batch_size_predict = args.batch_size_predict
voc_path = args.voc_path

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
                                    batch_size=batch_size_predict, build_voc = False,
                                    voc_path = voc_path)
     

    cnn = CnnTextClassifier()
    
    pred_dic = cnn.predict(predict_generator)
    
    plt.figure()
    
    plot_confusion_matrix(pred_dic['ground_truth'],pred_dic['predictions'] , classes=['positive','negative'],
                      title='Confusion matrix, without normalization')

    plt.figure()
    
    plot_confusion_matrix(pred_dic['ground_truth'],pred_dic['predictions'] , classes=['positive','negative'],
                      title='Confusion matrix, with normalization', normalize=True)
    plt.show()

if __name__ == '__main__':
    
    predict()
    
    