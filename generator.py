# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:22:17 2017

@author: Alexandre Boyker
"""
from vocab import VocBuilder
import numpy as np
from nltk.stem.porter import PorterStemmer
import codecs
import os
import json
from collections import deque
import traceback


class BatchGenerator(object):
    
    """
    Generates batches of vectorized sentences. The vectors generated are of the form
    [ 2, 34, 72, 0, 0, 0, ...,0] for instance. If the original sentences is 'I eat bread'.
    2 --> I , eat --> 34, bread --> 72 (voc mapping). The zero's are just padding tokens and the length of 
    these vectors equals the length of the longest sentence in the corpus.
    
    positional arguments for constructor:
       
    filenames -- list of .txt files path, which will be used to build the voc
    
    parser -- a parser object defined as shown in the template doc.
    
    Keyword argument for constructor:
        
    batch_size -- the number of samples to generate at each iteration (default = 32)
    
    stemmer -- the stemmer to use, should be the same as the one used to build the vocabulary
    (default = PorterStemmer() from nltk.stem.porter)
    
    build_voc -- if True, the voc is build from scratches. False means that the vocabulary has
    already been built
         
    
    """
    def __init__(self, filenames, parser, batch_size=32, stemmer = PorterStemmer(),  
                  max_sentence_length =None,
                 build_voc=False, voc_path='voc', voc_threshold=1):
        
        self.filenames = filenames
        self.parser = parser
        self.batch_size = batch_size
        self.stemmer = stemmer
        self.len_doc = 0
        self.voc_threshold = voc_threshold
        
        if build_voc:
            print(self.filenames)
            vb = VocBuilder( self.filenames, self.parser, voc_path=voc_path,
                            voc_threshold=self.voc_threshold, stemmer=stemmer)
            vb.build_vocab()
            #max_sentence_length = vb.max_sentence_length
            #print("max sentence length {}".format(max_sentence_length))
        
        voc_components = ['index2word.json', 'word2index.json', 'voc_summary.json' ]
        
        #sanity check to see if all vocab files are present
        try:
            for item in voc_components:       
                assert item in os.listdir(voc_path)
        except FileNotFoundError:
            raise Exception("voabulary has not been created, set build_voc = True in BatchGenerator contructor")
        
        # counts the total number of lines in the input documents
        for filename in self.filenames:   
            number_of_lines = 0
            f = codecs.open(filename, 'r', encoding = "utf8",errors = 'ignore')
            for line in f:
                number_of_lines += 1
            self.len_doc += number_of_lines
            
        if self.batch_size is None:
            
            self.batch_size =  self.len_doc
            
        print("loading voc data...")
        with open(os.path.join(os.getcwd(), voc_path,'index2word.json')) as data_file:    
            self.index2word = json.load(data_file)
        self.index2word[len(self.index2word)] ='<PAD>'

        with open(os.path.join(os.getcwd(), voc_path,'word2index.json')) as data_file:    
            self.word2index = json.load(data_file)
        self.word2index['<PAD>'] = 0
        with open(os.path.join(os.getcwd(), voc_path,'voc_summary.json')) as data_file:    
            self.voc_summary = json.load(data_file)
        if max_sentence_length is None:
            self.sequence_len = self.voc_summary['max_sequence_len']
        elif max_sentence_length > self.voc_summary['max_sequence_len'] :
            self.sequence_len = self.voc_summary['max_sequence_len']
        else:
            self.sequence_len = max_sentence_length
        
        self.vocab_size = len(self.index2word)
        print("...voc data loaded")

    def fetch_index(self, word):
        """ Returns the voc index associated to a stemmed token
        """
        try:
            return self.word2index[word]
        except: 
            return 0
        
    def __iter__(self):
        """Returns batches of vectorized sentences
        """
        for filename in self.filenames:
            
            while True:
                
                f = codecs.open(filename, 'r', encoding = "utf8",errors = 'ignore')
                enum_f = enumerate(f)
                batch_input = deque()
                batch_label = deque()
                
                for i,line in enum_f:
                            parsed_line = self.parser.parse_line(line)
                            txt = parsed_line[0].split()
                            seq_len = len(txt)
                            
                            if seq_len>=self.sequence_len:
                                txt = txt[:self.sequence_len]
                                
                            elif seq_len<self.sequence_len:
                                txt = txt + ['<PAD>' for i in range(self.sequence_len-seq_len)] 
                            tokens = [self.stemmer.stem(wrd)  if wrd!='<PAD>' else '<PAD>' for wrd in txt]
                            index_of_tokens = [self.fetch_index(word) for word in tokens ]
                            
                            
                            


                            batch_input.append(np.array(index_of_tokens))
                            batch_label.append(parsed_line[1])
                            if i==0:
                                 self.num_classes = len(set(parsed_line[1]))
                            
                            if len(batch_label)==self.batch_size:
                                yield (np.array(batch_input),  np.array(batch_label)) 
                                batch_input = deque()
                                batch_label = deque()
                            
                    
                      
