import numpy as np
import re

class StsaParser(object):
    def __init__(self):
        pass
    def transform_label_to_numeric(self, y):
            if '1' in y:
                return np.array([1,0])
            else:
                return np.array([0,1])


    def parse_line(self, row):
        
        row = row.split(' ')
        text = (' '.join(row[1:]))
        label = self.transform_label_to_numeric(row[0])
        return (re.sub(r'\W+', ' ', text), label)   
    
    

class AmazonParser(object):
    def __init__(self):
        pass
    def transform_label_to_numeric(self, y):
            if '2' in y:
                return np.array([1,0])
            else:
                return np.array([0,1])


    def parse_line(self, row):
        
        row = row.split(' ')
        text = (' '.join(row[1:]))
        label = self.transform_label_to_numeric(row[0])
        return (re.sub(r'\W+', ' ', text), label)       




class MovieParser(object):
    def __init__(self):
        
        self.error_cnt = 0
        
    def transform_label_to_numeric(self, y):
            
            if '1' in list(y):
                return np.array([1,0])
            else:
                return np.array([0,1])


    def parse_line(self, row):
        
        row = row.split('\t')
        text = ((row[0]))
        label = self.transform_label_to_numeric(row[1])
        
        return (re.sub(r'\W+', ' ', text), label) 

class MbtiParser(object):
    def __init__(self):
        pass
    def get_label(self,x):
        if x=='ISTJ':
            return np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        if x=='ISTP':
            return np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        if x=='ESTP':
            return np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
        if x=='ESTJ':
            return np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
        if x=='ISFJ':
            return np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
        if x=='ISFP':
            return np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
        if x=='ESFP':
            return np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        if x=='ESFJ':
            return np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
        if x=='INFJ':
            return np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
        if x=='INFP':
            return np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        if x=='ENFP':
            return np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
        if x=='ENFJ':
            return np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
        if x=='INTJ':
            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
        if x=='INTP':
            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
        if x=='ENTP':
            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
        if x=='ENTJ':
            return np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
        
    def parse_line(self, line):
        row = ((line.split(",")))
        #label = np.array([1,0]) if row[0][0]=='I' else np.array([0,1])
        label = self.get_label(row[0])
        row = ' '.join(row[1:])
        return (re.sub(r'\W+', ' ', row), label)
