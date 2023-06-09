# -*- coding: utf-8 -*-
"""
@author: jiyoon
"""

from matplotlib import type1font
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
from models.TCN_SNGP import *

from configs.configs import get_parser
from configs.tcn_sngp import tcn_sngp_parser
import dataloaders.data_loader as dr
import argparse

import warnings
warnings.filterwarnings('ignore')

class Train_Inference():
    def __init__(self, 
                 args: argparse):
        
        self.args = args
      
        self.data_loader = dr.data_loader(self.args)
        self.data_dict = self.data_loader.preprocessing()
        
        self.train_x, self.valid_x, self.test_x_t1, self.test_x_t2 = self.data_dict['X']
        self.train_y, self.valid_y, self.test_y_t1, self.test_y_t2 = self.data_dict['y']
        self.kc_enc, self.ukc_enc =  self.data_dict['enc']
        
        self.total_label = self.data_dict['total_label']
        self.unknwon_label = [self.total_label[-1]]
        self.known_label = self.total_label[:-1]
        
        _, self.time_steps, self.input_dim = self.train_x.shape 


    def Train(self):     
        model = compiled_tcn_sngp(num_feat=self.input_dim, num_classes=self.args.num_known, 
                                nb_filters=self.args.nb_filters, kernel_size=self.args.kernel_size, 
                                num_inducing=self.args.num_inducing, gp_cov_momentum=self.args.gp_cov_momentum, 
                                gp_cov_ridge_penalty=self.args.gp_cov_ridge_penalty, normalize_input=self.args.normalize_input,                               
                                scale_random_features=self.args.scale_random_features, gp_kernel_scale=self.args.gp_kernel_scale,
                                dilations=[1, 2, 4, 8, 16, 32, 64, 128], nb_stacks=self.args.nb_stacks, max_len=100, output_len=1, padding='causal',
                                use_skip_connections=True, return_sequences=False, regression=False,  # TODO: Comparison of use/non-use of skip-connection during comparison experiment
                                dropout_rate=self.args.dropout_rate, name='tcnsngp', kernel_initializer='he_normal', activation='relu', opt='adam', lr=self.args.lr, 
                                use_batch_norm=False, use_layer_norm=False, use_weight_norm=False, use_spectral_norm=True, norm_multiplier=self.args.norm_multiplier)
    
        model.summary()
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1E-5, patience=30, restore_best_weights=True)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        
        model.compile(tf.keras.optimizers.Adam(learning_rate=1E-3), loss=loss, metrics=['categorical_accuracy'])  
        model.fit(self.train_x, self.train_y, batch_size=128, epochs=self.args.epochs, validation_data=(self.valid_x, self.valid_y), callbacks = [early_stop])

        ckpt_dir = self.args.ckpt_dir
        
        self.save_name = f'{self.args.method}/{self.args.scenario}/{self.args.norm_multiplier}_{self.args.gp_kernel_scale}_{self.args.gp_cov_ridge_penalty}_{self.args.num_inducing}_{self.args.gp_cov_momentum}_{self.args.nb_stacks}/{self.args.noise_ratio}/known_class_{self.args.num_known}/unknown_cls_ind_{self.args.unknown_cls_idx}'
        self.model_ckpt_dir = os.path.join(ckpt_dir, self.save_name)
        os.makedirs(self.model_ckpt_dir, exist_ok=True)
        model.save(self.model_ckpt_dir)
        return 
    
    def Inference_train(self, threshold):
        self.threshold=threshold
        
        model = tf.keras.models.load_model(self.model_ckpt_dir)
            
        logits = model.predict(self.train_x)
        probs = tf.nn.softmax(logits, axis=-1).numpy()

        y_pred = np.argmax(probs, axis=1)
        train_y = self.kc_enc.inverse_transform(self.train_y)
        cm = confusion_matrix(train_y, y_pred) 
        
        save_dir = self.args.save_dir     
        self.plot_save_dir = os.path.join(save_dir, self.save_name)
        os.makedirs(self.plot_save_dir, exist_ok=True)
        
        self.plot_confusion_matrix(cm, labels=self.known_label, save_name='only_known_train') 
    
        logits_df = pd.DataFrame(logits, columns=self.known_label)
        labels = pd.DataFrame(train_y, columns=['LABEL'])
        train_logits = pd.concat([logits_df, labels], axis=1)
        train_logits.to_csv(os.path.join(self.plot_save_dir,'train_logit.csv'), index=False)
        
        tr_sum_logits = tf.math.exp(logits_df)
        tr_sum_logits = tf.reduce_sum(tr_sum_logits, axis=1).numpy()
        tr_uncertainty = self.args.num_known/(self.args.num_known+tr_sum_logits)
        
        self.tr_threshold = np.percentile(tr_uncertainty, threshold)
        return
        
    def Inference_test(self, type = 1):
        model = tf.keras.models.load_model(self.model_ckpt_dir)

        if type == 1:
            test_x = self.test_x_t1
            test_y = self.test_y_t1
        
        elif type == 2:
            test_x = self.test_x_t2
            test_y = self.test_y_t2
            
        test_logits = model.predict(test_x)
        test_probs = tf.nn.softmax(test_logits, axis=-1).numpy()
        
        '''
        Prediction using only probability
        '''
        y_test_pred = np.argmax(test_probs, axis=1)
        test_y = self.ukc_enc.inverse_transform(test_y)
        cm = confusion_matrix(test_y, y_test_pred)
        
        self.plot_confusion_matrix(cm, labels=self.total_label, save_name=f'only_known_test_type{type}') 
         
        test_logits_df = pd.DataFrame(test_logits, columns=self.known_label)
        test_labels = pd.DataFrame(test_y, columns=['LABEL'])
        test_logits = pd.concat([test_logits_df, test_labels], axis=1)
        test_logits.to_csv(os.path.join(self.plot_save_dir,f'test_type{type}_logit.csv'), index=False)
                        
        '''
        Prediction using with uncertainty 
        '''            
        sum_logits = tf.math.exp(test_logits_df) # 오타 수정
        sum_logits = tf.reduce_sum(sum_logits, axis=1).numpy()
        uncertainty = self.args.num_known/(self.args.num_known+sum_logits)
        
        pd.DataFrame(test_probs).to_csv(os.path.join(self.plot_save_dir, f'prob_test_type{type}.csv'), index=False)
        test_pred = pd.DataFrame(y_test_pred, columns=['PRED'])
        test_true = pd.DataFrame(test_y, columns=['TRUE'])
        test_table = pd.concat((test_true, test_pred), axis=1)
        test_table.to_csv(os.path.join(self.plot_save_dir, f'table_test_type{type}.csv'), index=False)

        test_pred_uncertainty = np.argmax(test_probs, axis=1)
        test_pred_uncertainty[uncertainty>=self.tr_threshold] = self.args.num_known
        
        cm = confusion_matrix(test_y, test_pred_uncertainty)
        self.plot_confusion_matrix(cm, labels=self.total_label, save_name=f'openset_thres_{self.threshold}_type{type}')
        
        cr = np.round(pd.DataFrame(classification_report(test_y, test_pred_uncertainty, output_dict=True)), 4)
        cr.columns = self.total_label+['accuracy', 'macro avg','weighted avg']
        
        known_metric = np.average(cr.iloc[:3,:self.args.num_known], axis=1)
        cr['known'] = list(known_metric) + [np.nan]            
        
        cr['avg'] = cr.iloc[:3, self.args.num_known]
        cr['avg'] = cr[['avg', 'known']].mean(axis=1)      
        cr.to_csv(os.path.join(self.plot_save_dir,f'results_{self.threshold}_type{type}.csv'))
            
        return  
        
    def plot_confusion_matrix(self, con_mat, labels, save_name='only_known', title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
        plt.figure(figsize=(8,6))
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title, fontsize=11)
        plt.colorbar(fraction=0.046, pad=0.04)
        
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(con_mat)):
            n = sum(con_mat[k])
            nlabel = '{0} (n={1})'.format(labels[k],n)
            nlabels.append(nlabel)
        plt.xticks(marks, labels, fontsize=10)
        plt.xticks(rotation=90)
        plt.yticks(marks, nlabels, fontsize=10)

        thresh = con_mat.max() / 2.
        if normalize:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        else:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label\n', fontsize=11)
        plt.xlabel('\nPredicted label', fontsize=11)

        plt.savefig(os.path.join(self.plot_save_dir, f'TCN_SNGP_{save_name}.png'))
        plt.clf()    
