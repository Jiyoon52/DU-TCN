# -*- coding: utf-8 -*-
"""
@author: jiyoon
"""

import tensorflow as tf

import pickle
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt     
import itertools
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from configs.configs import get_parser
from configs.tcn_sngp import tcn_sngp_parser
import argparse
 
class data_loader():
    
    def __init__(self, 
                 args: argparse):
        
        self.args = args
        # self.data_load_dict = self.preprocessing()
        
    def preprocessing(self):
            
        print('Experiment Info...')
        print(f'Seed                 : {self.args.seed}')
        print(f'Scenario             : {self.args.scenario}')
        print(f'Num of known Classes : {self.args.num_known}')
        
        data_dir = os.path.join(self.args.data_dir, f'{self.args.scenario}')
            
        kc_enc = OneHotEncoder(sparse=False, categories='auto')
        ukc_enc = OneHotEncoder(sparse=False, categories='auto')
        
        enc = OneHotEncoder(sparse=False, categories='auto')
        enc_ = np.concatenate([np.repeat(i, 1000) for i in range(self.args.num_known+1)]).reshape(-1, 1) # num of unknown = 1
        enc_ = enc.fit_transform(enc_)
    
        noise_ratio = self.args.noise_ratio # Set the size of assigned noise
        data_origin_dir = os.path.join(data_dir, '0.02') # Small noise
        data_scenario_dir = os.path.join(data_dir, noise_ratio) # Large noise
        
        '''
        Train data w/ noise ratio = 0.02 (from data_origin_dir)
        '''
        Train_X_dir = os.path.join(data_origin_dir, 'Train_X.pkl')
        with open(Train_X_dir, 'rb') as f:
            train_x = pickle.load(f)
        
        train_x = train_x[:1000*(self.args.num_known+1), :, :(self.args.num_known+1)]
         
        Train_y_dir = os.path.join(data_origin_dir, 'Train_y.pkl')
        with open(Train_y_dir, 'rb') as f:
            train_y = pickle.load(f)
            
        train_y = train_y[:1000*(self.args.num_known+1), :self.args.num_known+1]
        train_y = enc.inverse_transform(train_y).flatten()
    
        '''
        Type 1
        
        Test data w/ noise ratio = 0.02 (from data_origin_dir)
        '''
        # Type 1
        Test_X_dir = os.path.join(data_origin_dir, 'Test_X.pkl')
        with open(Test_X_dir, 'rb') as f:
            test_x_t1 = pickle.load(f)
            
        test_x_t1 = test_x_t1[:200*(self.args.num_known+1), :, :(self.args.num_known+1)]
             
        Test_y_dir = os.path.join(data_origin_dir, 'Test_y.pkl')
        with open(Test_y_dir, 'rb') as f:
            test_y_t1 = pickle.load(f)
                
        test_y_t1 = test_y_t1[:200*(self.args.num_known+1), :self.args.num_known+1]
        test_y_t1 = enc.inverse_transform(test_y_t1).flatten()
       
        '''
        Type 2
        
        Test data w/ noise ratio = self.args.noise_ratio (from data_scenario_dir)
        ''' 
        # Type 2
        Test_X_dir = os.path.join(data_scenario_dir, 'Test_X.pkl')
        with open(Test_X_dir, 'rb') as f:
            test_x_t2 = pickle.load(f)
            
        test_x_t2 = test_x_t2[:200*(self.args.num_known+1), :, :(self.args.num_known+1)]
             
        Test_y_dir = os.path.join(data_scenario_dir, 'Test_y.pkl')
        with open(Test_y_dir, 'rb') as f:
            test_y_t2 = pickle.load(f)
                
        test_y_t2 = test_y_t2[:200*(self.args.num_known+1), :self.args.num_known+1]
        test_y_t2 = enc.inverse_transform(test_y_t2).flatten()
        
        print('Dims of train dataset')
        print(train_x.shape) #shape : (num_of_instance x window_size x input_dims) = (4000, 100, 4)
        print(train_y.shape) #shape : (num_of_instance) = (4000,)
        print('--'*15)
        print('Dims of type 1 dataset')
        print(test_x_t1.shape)  #shape : (num_of_instance x window_size x input_dims) = (800, 100, 4)
        print(test_y_t1.shape)  #shape : (num_of_instance) = (800,)
        print('--'*15) 
        print('Dims of type 2 dataset')
        print(test_x_t2.shape)  #shape : (num_of_instance x window_size x input_dims) = (800, 100, 4)
        print(test_y_t2.shape)  #shape : (num_of_instance) = (800,)

    
        '''
        set unknown class
        '''
        
        unknown_cls_idx = self.args.unknown_cls_idx # can change this setting (It means #4 class assigned as unknown class)
        train_y[train_y==unknown_cls_idx] = len(set(train_y))
        train_y[train_y>unknown_cls_idx] = train_y[train_y>unknown_cls_idx]-1
    
        test_y_t1[test_y_t1==unknown_cls_idx] = len(set(train_y))
        test_y_t1[test_y_t1>unknown_cls_idx] = test_y_t1[test_y_t1>unknown_cls_idx]-1
    
        test_y_t2[test_y_t2==unknown_cls_idx] = len(set(train_y))
        test_y_t2[test_y_t2>unknown_cls_idx] = test_y_t2[test_y_t2>unknown_cls_idx]-1
    
        train_x = train_x[train_y != len(set(test_y_t1))-1] #(3000, 100, 4)
        train_y = train_y[train_y != len(set(test_y_t1))-1] #(3000,)
          
        known_cls_set = set(train_y)
    
        print('Dims of train dataset')
        print(train_x.shape) #shape : (num_of_instance x window_size x input_dims) = (3000, 100, 4) # Num of Class = 3
        print(train_y.shape) #shape : (num_of_instance) = (3000,)
        print(set(train_y))
        print('--'*15)
        print('Dims of type 1 dataset')
        print(test_x_t1.shape)  #shape : (num_of_instance x window_size x input_dims) = (800, 100, 4) # Num of Class = 4
        print(test_y_t1.shape)  #shape : (num_of_instance) = (800,)
        print(set(test_y_t1))
        print('--'*15) 
        print('Dims of type 2 dataset')
        print(test_x_t2.shape)  #shape : (num_of_instance x window_size x input_dims) = (800, 100, 4) # Num of Class = 4
        print(test_y_t2.shape)  #shape : (num_of_instance) = (800,)
        print(set(test_y_t2))
        
        train_y = kc_enc.fit_transform(train_y.reshape(-1,1))
        test_y_t1 = ukc_enc.fit_transform(test_y_t1.reshape(-1,1))      
        test_y_t2 = ukc_enc.fit_transform(test_y_t2.reshape(-1,1))      
    
        print('Dims of encoded y')
        print(train_y.shape)  
        print(test_y_t1.shape) 
        print(test_y_t2.shape) 
    
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, stratify=train_y, random_state=self.args.seed)
       
        gt_label = [i for i in range(self.args.num_known+1)]
        total_label = [i for i in range(self.args.num_known+1)]
        known_label = [i for i in range(self.args.num_known+1)]
        known_label.remove(total_label[unknown_cls_idx])
        known_label.extend([f'{total_label[unknown_cls_idx]}(Unknown)'])
        
       
        
        total_label = known_label
        unknwon_label = [total_label[-1]]
        known_label = total_label[:-1]
        
        print(f'total label is {total_label}')
        print(f'unknwon label is {unknwon_label}')
        print(f'known label is {known_label}')
        
        
        return dict(X = (train_x, valid_x, test_x_t1, test_x_t2), y = (train_y, valid_y, test_y_t1, test_y_t2),
                    enc = (kc_enc, ukc_enc), total_label = total_label)
    
