# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:08:36 2017

@author: Alexandre Boyker
"""
import os
import codecs
from nltk.stem.porter import PorterStemmer
import json
from datetime import datetime


class VocBuilder(object):
    """This class can be used to build a vocabulary, given a list of filenames (text documents)
    
    positional arguments for constructor:
       
    filenames -- list of .txt files path, which will be used to build the voc
    
    parser -- a parser object defined as shown in the template doc. 
        
    Keyword argument for constructor:
        
    voc_path -- the directory where the voc files will be stored (default='voc_path' )
    
    voc_threshold -- the minimum number of occurence required for a term
    to be taken into account in the vocabulary(default=1)
    
    max_line -- the maximum number of lines to consider when building the voc.
    This can be useful to limit the number of lines for large corpus. For instance, 
    Amazon review corpus contains >3.5 million lines. Building a vocabulary with the
    first 1 million line is sufficient.
    
    """
    def __init__(self, filenames, parser, voc_path='voc', 
                 voc_threshold = 1, max_line =None, stemmer = PorterStemmer()):
        
        self.max_sentence_length = 0
        self.voc_path =voc_path
        self.voc_threshold = voc_threshold
        self.filenames = filenames
        self.parser = parser
        self.max_line = max_line
        self.stemmer = stemmer
        # create directory to store voc files
        if not os.path.exists(self.voc_path):
            os.makedirs(self.voc_path)
    
        
    def build_vocab(self):
        index_voc = 1
        word2index = {}
        index2word = {}
        voc_summary = {}
        word_counter = {}
        for filename in self.filenames:
            
            f = codecs.open(filename, 'r', encoding = "utf8", errors = 'ignore')
            enum_f = enumerate(f)
            try:
                for ind,line in enum_f:
                    parsed_line = self.parser.parse_line(line)
                    tokens = [(self.stemmer.stem(wrd)) for wrd in parsed_line[0].split()]
                    l_tokens = len(tokens)
                    if l_tokens > self.max_sentence_length:
                        self.max_sentence_length = l_tokens
                    for word in tokens:
                        word_counter[word] = word_counter[word] + 1 if word in word_counter else 1
                        if word_counter[word]==self.voc_threshold :
                            index2word[index_voc] = word
                            word2index[word] = index_voc
                            index_voc += 1
                    if ind%1000 ==0: print("{}: {} lines processed".format(str(datetime.now()),ind) )
                    if ind == self.max_line: break
            except Exception as e:
                pass
        
        voc_summary['max_sequence_len'] = self.max_sentence_length
        voc_summary['voc_len'] = len(index2word)
        
        with open(os.path.join(os.getcwd(),self.voc_path,'voc_summary.json'), 'w') as outfile:
                        json.dump(voc_summary, outfile)
                        
        
        with open(os.path.join(os.getcwd(), self.voc_path,'index2word.json'), 'w') as outfile:
                        json.dump(index2word, outfile)
        
        with open(os.path.join(os.getcwd(),self.voc_path,'word2index.json'), 'w') as outfile:
                        json.dump(word2index, outfile)

