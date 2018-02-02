""" Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length. This example is using
a toy dataset to classify linear sequences. The generated sequences have
variable length.
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
import random
import numpy as np
import re
import rnn_tianyiluo
import scipy
from scipy import spatial
# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
                 max_value=1000):
        
        def cal_simi_twovecs(fun_temp_most_relevant_view_event_words, fun_temp_item):
            #vector_one = np.array(fun_temp_most_relevant_view_event_words)
            vector_one = np.asarray(fun_temp_most_relevant_view_event_words, dtype=float)
            #vector_two = np.array(fun_temp_item)
            vector_two = np.asarray(fun_temp_item, dtype=float)
            temp_distance = spatial.distance.cosine(vector_one, vector_two)
            return temp_distance
        
        self.search_event_words_list = []
        self.search_event_index_list = []
        self.view_event_words_list = []
        self.view_event_index_list = []
        
        self.data_train = []
        self.labels_train = []
        self.seqlen_train = []
        self.relevantkey_search_train = []
        self.relevantvalue_search_train = []
        self.relevantkey_view_train = []
        self.relevantvalue_view_train = []
        
        self.data_test = []
        self.labels_test = []
        self.seqlen_test = []
        self.relevantkey_search_test = []
        self.relevantvalue_search_test = []
        self.relevantkey_view_test = []
        self.relevantvalue_view_test = []
        
        #get userfeature + ad corpus
        simi_threshold = 0.5
        userid_dic = {}
        finish_process_dic = {}
        num_different_userid = 0
        max_len_sequence = 40
        num_features = 16 + 20
        vector_zero = []
        for i_zero_index in range(num_features):
            vector_zero.append(0.0)
        #vector_zero = np.array(vector_zero)
        #attention_userplusad_file = "tensorflow_sample_100000_newid.csv"
        #attention_userplusad_file = "tensorflow_train_sample_100000_20171104_newid.csv"
        attention_userplusad_file = "purchased_end_tensorflow_train_sample_100000_20171130_newid_combined.csv"
        trainX = []
        trainY = []
        testX = []
        testY = []
        current_sequence_list = []
        current_feature_list = []
        current_num_display_ad = 0
        line_num = 0
        train_num_point = 160000
        haha_num = 0
        last_userid = "1"
        for line_original in open(attention_userplusad_file, 'r'):
            line_num += 1
            if line_num % 1000000 == 0:
                print(line_num)
            #if line_num > 32000:
            #    break
            line = re.sub(" +", " ", line_original)
            line_str = line.strip().split(" ")
            #print(len(line_str))
            if len(line_str) != 16 + 3 and len(line_str) != 3 + num_features:
                continue
            if len(line_str) != 3 + num_features:
                #print(line_num)
                #print(len(line_str))
                #print(line_str)
                for i in range(num_features - 16):
                    line_str.append("0.0")
                temp_label = line_str[16 + 2]
                line_str[16 + 2] = line_str[num_features + 2]
                line_str[num_features + 2] = temp_label
            temp_userid = line_str[0]
            #see if we finish processing this session
            if temp_userid in finish_process_dic:
                continue
            if temp_userid not in userid_dic:
                userid_dic[temp_userid] = temp_userid
                haha_num += 1
                num_different_userid += 1
                if current_num_display_ad != 0:
                    finish_process_dic[last_userid] = last_userid
                    remian_num_display_ad = max_len_sequence - current_num_display_ad
                    for i_temp in range(remian_num_display_ad):
                        current_sequence_list.append(vector_zero)
                    
                    if num_different_userid - 1 <= train_num_point:
                        self.seqlen_train.append(current_num_display_ad)
                        
                        if len(self.search_event_words_list) == 0:
                            self.relevantkey_search_train.append(0)
                            self.relevantvalue_search_train.append(0)
                        else:
                            final_initial_search = self.search_event_index_list[0]
                            temp_most_relevant_search_event_words = self.search_event_words_list[len(self.search_event_words_list) - 1]
                            for index_list in range(len(self.search_event_words_list)):
                                if index_list == len(self.search_event_words_list) - 1:
                                    continue
                                temp_item = self.search_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_search = self.search_event_index_list[index_list]
                                    break
                            if final_initial_search == self.search_event_index_list[0]:
                                self.relevantkey_search_train.append(0)
                                self.relevantvalue_search_train.append(0) 
                            else:
                                self.relevantkey_search_train.append(final_initial_search)
                                self.relevantvalue_search_train.append(len(self.search_event_words_list) - 1)
                        
                        if len(self.view_event_words_list) == 0:
                            self.relevantkey_view_train.append(0)
                            self.relevantvalue_view_train.append(0)                        
                        else:
                            final_initial_view = self.view_event_index_list[0]
                            temp_most_relevant_view_event_words = self.view_event_words_list[len(self.view_event_words_list) - 1]
                            for index_list in range(len(self.view_event_words_list)):
                                if index_list == len(self.view_event_words_list) - 1:
                                    continue
                                temp_item = self.view_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_view = self.view_event_index_list[index_list]
                                    break
                            if final_initial_view == self.view_event_index_list[0]:
                                self.relevantkey_view_train.append(0)
                                self.relevantvalue_view_train.append(0) 
                            else:
                                self.relevantkey_view_train.append(final_initial_view)
                                self.relevantvalue_view_train.append(len(self.view_event_words_list) - 1)
                        
                        self.data_train.append(current_sequence_list)
                        #print(line_num)
                        if line_str[num_features + 2] == '1':               
                            self.labels_train.append([1.0, 0.0])
                        else:
                            self.labels_train.append([0.0, 1.0])
                        current_sequence_list = []
                        current_num_display_ad = 0
                    else:
                        self.seqlen_test.append(current_num_display_ad)
                        
                        if len(self.search_event_words_list) == 0:
                            self.relevantkey_search_test.append(0)
                            self.relevantvalue_search_test.append(0)
                        else:
                            final_initial_search = self.search_event_index_list[0]
                            temp_most_relevant_search_event_words = self.search_event_words_list[len(self.search_event_words_list) - 1]
                            for index_list in range(len(self.search_event_words_list)):
                                if index_list == len(self.search_event_words_list) - 1:
                                    continue
                                temp_item = self.search_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_search = self.search_event_index_list[index_list]
                                    break
                            if final_initial_search == self.search_event_index_list[0]:
                                self.relevantkey_search_test.append(0)
                                self.relevantvalue_search_test.append(0) 
                            else:
                                self.relevantkey_search_test.append(final_initial_search)
                                self.relevantvalue_search_test.append(len(self.search_event_words_list) - 1)
                        
                        if len(self.view_event_words_list) == 0:
                            self.relevantkey_view_test.append(0)
                            self.relevantvalue_view_test.append(0)         
                        else:
                            final_initial_view = self.view_event_index_list[0]
                            temp_most_relevant_view_event_words = self.view_event_words_list[len(self.view_event_words_list) - 1]
                            for index_list in range(len(self.view_event_words_list)):
                                if index_list == len(self.view_event_words_list) - 1:
                                    continue
                                temp_item = self.view_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_view = self.view_event_index_list[index_list]
                                    break
                            if final_initial_view == self.view_event_index_list[0]:
                                self.relevantkey_view_test.append(0)
                                self.relevantvalue_view_test.append(0) 
                            else:
                                self.relevantkey_view_test.append(final_initial_view)
                                self.relevantvalue_view_test.append(len(self.view_event_words_list) - 1)
                        
                        #self.relevantkey_test.append(1)
                        #self.relevantvalue_test.append(int(current_num_display_ad / 2) + 1)
                        #current_sequence_list = np.array(current_sequence_list)
                        self.data_test.append(current_sequence_list)
                        if line_str[num_features + 2] == '1':
                            self.labels_test.append([1.0, 0.0])
                        else:
                            self.labels_test.append([0.0, 1.0])
                        current_sequence_list = []
                        current_num_display_ad = 0
                    #sucessful store one user, begin another user
                    self.search_event_words_list = []
                    self.search_event_index_list = []
                    self.view_event_words_list = []
                    self.view_event_index_list = []
                    current_num_display_ad += 1
                    for i in range(num_features):
                        current_feature_list.append(float(line_str[i + 1]))
                    #current_feature_list = np.array(current_feature_list)
                    current_sequence_list.append(current_feature_list)
                    current_feature_list = []
                    if line_str[3] == '1':
                        self.search_event_words_list.append(line_str[18:38])
                        self.search_event_index_list.append(current_num_display_ad - 1)
                    if line_str[1] == '1':
                        self.view_event_words_list.append(line_str[18:38])
                        self.view_event_index_list.append(current_num_display_ad - 1)
                else:# just for the first line
                    current_num_display_ad += 1
                    for i in range(num_features):
                        current_feature_list.append(float(line_str[i + 1]))
                    #current_feature_list = np.array(current_feature_list)
                    current_sequence_list.append(current_feature_list)
                    current_feature_list = []
                    if line_str[3] == '1':
                        self.search_event_words_list.append(line_str[18:38])
                        self.search_event_index_list.append(current_num_display_ad - 1)
                    if line_str[1] == '1':
                        self.view_event_words_list.append(line_str[18:38])
                        self.view_event_index_list.append(current_num_display_ad - 1)
            else:
                current_num_display_ad += 1
                for i in range(num_features):
                    current_feature_list.append(float(line_str[i + 1]))
                #current_feature_list = np.array(current_feature_list)
                current_sequence_list.append(current_feature_list)
                
                if line_str[3] == '1':
                    self.search_event_words_list.append(line_str[18:38])
                    self.search_event_index_list.append(current_num_display_ad - 1)
                if line_str[1] == '1':
                    self.view_event_words_list.append(line_str[18:38])
                    self.view_event_index_list.append(current_num_display_ad - 1)
                
                if current_num_display_ad == max_len_sequence:
                    finish_process_dic[temp_userid] = temp_userid
                    if num_different_userid - 1 <= train_num_point:
                        #current_sequence_list = np.array(current_sequence_list)
                        self.seqlen_train.append(current_num_display_ad)
                        if len(self.search_event_words_list) == 0:
                            self.relevantkey_search_train.append(0)
                            self.relevantvalue_search_train.append(0)
                        else:
                            final_initial_search = self.search_event_index_list[0]
                            temp_most_relevant_search_event_words = self.search_event_words_list[len(self.search_event_words_list) - 1]
                            for index_list in range(len(self.search_event_words_list)):
                                if index_list == len(self.search_event_words_list) - 1:
                                    continue
                                temp_item = self.search_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_search = self.search_event_index_list[index_list]
                                    break
                            if final_initial_search == self.search_event_index_list[0]:
                                self.relevantkey_search_train.append(0)
                                self.relevantvalue_search_train.append(0) 
                            else:
                                self.relevantkey_search_train.append(final_initial_search)
                                self.relevantvalue_search_train.append(len(self.search_event_words_list) - 1)
                        
                        if len(self.view_event_words_list) == 0:
                            self.relevantkey_view_train.append(0)
                            self.relevantvalue_view_train.append(0)                        
                        else:
                            final_initial_view = self.view_event_index_list[0]
                            temp_most_relevant_view_event_words = self.view_event_words_list[len(self.view_event_words_list) - 1]
                            for index_list in range(len(self.view_event_words_list)):
                                if index_list == len(self.view_event_words_list) - 1:
                                    continue
                                temp_item = self.view_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_view = self.view_event_index_list[index_list]
                                    break
                            if final_initial_view == self.view_event_index_list[0]:
                                self.relevantkey_view_train.append(0)
                                self.relevantvalue_view_train.append(0) 
                            else:
                                self.relevantkey_view_train.append(final_initial_view)
                                self.relevantvalue_view_train.append(len(self.view_event_words_list) - 1)
                        
                        self.data_train.append(current_sequence_list)
                        if line_str[num_features + 2] == '1':
                            self.labels_train.append([1.0, 0.0])
                        else:
                            self.labels_train.append([0.0, 1.0])
                        current_sequence_list = []
                        current_num_display_ad = 0
                    else:
                        #current_sequence_list = np.array(current_sequence_list)
                        self.seqlen_test.append(current_num_display_ad)
                        if len(self.search_event_words_list) == 0:
                            self.relevantkey_search_test.append(0)
                            self.relevantvalue_search_test.append(0)
                        else:
                            final_initial_search = self.search_event_index_list[0]
                            temp_most_relevant_search_event_words = self.search_event_words_list[len(self.search_event_words_list) - 1]
                            for index_list in range(len(self.search_event_words_list)):
                                if index_list == len(self.search_event_words_list) - 1:
                                    continue
                                temp_item = self.search_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_search = self.search_event_index_list[index_list]
                                    break
                            if final_initial_search == self.search_event_index_list[0]:
                                self.relevantkey_search_test.append(0)
                                self.relevantvalue_search_test.append(0) 
                            else:
                                self.relevantkey_search_test.append(final_initial_search)
                                self.relevantvalue_search_test.append(len(self.search_event_words_list) - 1)
                        
                        if len(self.view_event_words_list) == 0:
                            self.relevantkey_view_test.append(0)
                            self.relevantvalue_view_test.append(0)         
                        else:
                            final_initial_view = self.view_event_index_list[0]
                            temp_most_relevant_view_event_words = self.view_event_words_list[len(self.view_event_words_list) - 1]
                            for index_list in range(len(self.view_event_words_list)):
                                if index_list == len(self.view_event_words_list) - 1:
                                    continue
                                temp_item = self.view_event_words_list[index_list]
                                temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                                if temp_similarity >= simi_threshold:
                                    final_initial_view = self.view_event_index_list[index_list]
                                    break
                            if final_initial_view == self.view_event_index_list[0]:
                                self.relevantkey_view_test.append(0)
                                self.relevantvalue_view_test.append(0) 
                            else:
                                self.relevantkey_view_test.append(final_initial_view)
                                self.relevantvalue_view_test.append(len(self.view_event_words_list) - 1)
                        self.data_test.append(current_sequence_list)
                        if line_str[num_features + 2] == '1':
                            self.labels_test.append([1.0, 0.0])
                        else:
                            self.labels_test.append([0.0, 1.0])
                        current_sequence_list = []
                        current_num_display_ad = 0
                current_feature_list = []
                last_userid = temp_userid
        #process the last userid
        if current_num_display_ad != 0:
            remian_num_display_ad = max_len_sequence - current_num_display_ad
            for i_temp in range(remian_num_display_ad):
                current_sequence_list.append(vector_zero)
            
            if line_str[3] == '1':
                self.search_event_words_list.append(line_str[18:38])
                self.search_event_index_list.append(current_num_display_ad - 1)
            if line_str[1] == '1':
                self.view_event_words_list.append(line_str[18:38])
                self.view_event_index_list.append(current_num_display_ad - 1)
            
            if num_different_userid - 1 <= train_num_point:
                #current_sequence_list = np.array(current_sequence_list)
                self.seqlen_train.append(current_num_display_ad)
                if len(self.search_event_words_list) == 0:
                    self.relevantkey_search_train.append(0)
                    self.relevantvalue_search_train.append(0)
                else:
                    final_initial_search = self.search_event_index_list[0]
                    temp_most_relevant_search_event_words = self.search_event_words_list[len(self.search_event_words_list) - 1]
                    for index_list in range(len(self.search_event_words_list)):
                        if index_list == len(self.search_event_words_list) - 1:
                            continue
                        temp_item = self.search_event_words_list[index_list]
                        temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                        if temp_similarity >= simi_threshold:
                            final_initial_search = self.search_event_index_list[index_list]
                            break
                    if final_initial_search == self.search_event_index_list[0]:
                        self.relevantkey_search_train.append(0)
                        self.relevantvalue_search_train.append(0) 
                    else:
                        self.relevantkey_search_train.append(final_initial_search)
                        self.relevantvalue_search_train.append(len(self.search_event_words_list) - 1)
                
                if len(self.view_event_words_list) == 0:
                    self.relevantkey_view_train.append(0)
                    self.relevantvalue_view_train.append(0)                        
                else:
                    final_initial_view = self.view_event_index_list[0]
                    temp_most_relevant_view_event_words = self.view_event_words_list[len(self.view_event_words_list) - 1]
                    for index_list in range(len(self.view_event_words_list)):
                        if index_list == len(self.view_event_words_list) - 1:
                            continue
                        temp_item = self.view_event_words_list[index_list]
                        temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                        if temp_similarity >= simi_threshold:
                            final_initial_view = self.view_event_index_list[index_list]
                            break
                    if final_initial_view == self.view_event_index_list[0]:
                        self.relevantkey_view_train.append(0)
                        self.relevantvalue_view_train.append(0) 
                    else:
                        self.relevantkey_view_train.append(final_initial_view)
                        self.relevantvalue_view_train.append(len(self.view_event_words_list) - 1)
                
                self.data_train.append(current_sequence_list)
                if line_str[num_features + 2] == '1':
                    self.labels_train.append([1.0, 0.0])
                else:
                    self.labels_train.append([0.0, 1.0])
                current_sequence_list = []
                current_num_display_ad = 0
            else:
                #current_sequence_list = np.array(current_sequence_list)
                self.seqlen_test.append(current_num_display_ad)

                if len(self.search_event_words_list) == 0:
                    self.relevantkey_search_test.append(0)
                    self.relevantvalue_search_test.append(0)
                else:
                    final_initial_search = self.search_event_index_list[0]
                    temp_most_relevant_search_event_words = self.search_event_words_list[len(self.search_event_words_list) - 1]
                    for index_list in range(len(self.search_event_words_list)):
                        if index_list == len(self.search_event_words_list) - 1:
                            continue
                        temp_item = self.search_event_words_list[index_list]
                        temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                        if temp_similarity >= simi_threshold:
                            final_initial_search = self.search_event_index_list[index_list]
                            break
                    if final_initial_search == self.search_event_index_list[0]:
                        self.relevantkey_search_test.append(0)
                        self.relevantvalue_search_test.append(0) 
                    else:
                        self.relevantkey_search_test.append(final_initial_search)
                        self.relevantvalue_search_test.append(len(self.search_event_words_list) - 1)
                
                if len(self.view_event_words_list) == 0:
                    self.relevantkey_view_test.append(0)
                    self.relevantvalue_view_test.append(0)         
                else:
                    final_initial_view = self.view_event_index_list[0]
                    temp_most_relevant_view_event_words = self.view_event_words_list[len(self.view_event_words_list) - 1]
                    for index_list in range(len(self.view_event_words_list)):
                        if index_list == len(self.view_event_words_list) - 1:
                            continue
                        temp_item = self.view_event_words_list[index_list]
                        temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                        if temp_similarity >= simi_threshold:
                            final_initial_view = self.view_event_index_list[index_list]
                            break
                    if final_initial_view == self.view_event_index_list[0]:
                        self.relevantkey_view_test.append(0)
                        self.relevantvalue_view_test.append(0) 
                    else:
                        self.relevantkey_view_test.append(final_initial_view)
                        self.relevantvalue_view_test.append(len(self.view_event_words_list) - 1)

                self.data_test.append(current_sequence_list)
                if line_str[num_features + 2] == '1':
                    self.labels_test.append([1.0, 0.0])
                else:
                    self.labels_test.append([0.0, 1.0])
                current_sequence_list = []
                current_num_display_ad = 0
        
        self.batch_id = 0
        print("Initializing finished!!!")
        print(haha_num)

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data_train) or len(self.data_train) - self.batch_id < batch_size:
            self.batch_id = 0
        batch_data = (self.data_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        batch_labels = (self.labels_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        batch_seqlen = (self.seqlen_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        batch_key_search = (self.relevantkey_search_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        batch_key_view = (self.relevantvalue_search_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        batch_value_search = (self.relevantkey_view_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        batch_value_view = (self.relevantvalue_view_train[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data_train))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data_train))
        return batch_data, batch_labels, batch_seqlen, batch_key_search, batch_key_view, batch_value_search, batch_value_view


# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 40000
batch_size = 100
display_step = 1000

# Network Parameters
seq_max_len =40 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not

trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
#testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 36])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])
relevantevent_search_key = tf.placeholder(tf.int32, [None])
relevantevent_search_value = tf.placeholder(tf.int32, [None])
relevantevent_view_key = tf.placeholder(tf.int32, [None])
relevantevent_view_value = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, relevantevent_search_key, relevantevent_search_value, relevantevent_view_key, relevantevent_view_value, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_tianyiluo.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = rnn_tianyiluo.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen, relevanteventsearchkey=relevantevent_search_key, relevanteventsearchvalue=relevantevent_search_value, relevanteventviewkey=relevantevent_view_key, relevanteventviewvalue=relevantevent_view_value)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = dynamicRNN(x, seqlen, relevantevent_search_key, relevantevent_search_value, relevantevent_view_key, relevantevent_view_value, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
model_path = "/home/shaotang/Desktop/eclipseworkspace/deep_learning_ebay_similarity_items/20170129_middle_model"

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    
    #define the test dataset
    test_data = trainset.data_test
    test_label = trainset.labels_test
    test_seqlen = trainset.seqlen_test
    
    train_relevant_search_event_key = trainset.relevantkey_search_train
    train_relevant_search_event_value = trainset.relevantvalue_search_train
    train_relevant_view_event_key = trainset.relevantkey_view_train
    train_relevant_view_event_value = trainset.relevantvalue_view_train
    
    test_relevant_search_event_key = trainset.relevantkey_search_test
    test_relevant_search_event_value = trainset.relevantvalue_search_test
    test_relevant_view_event_key = trainset.relevantkey_view_test
    test_relevant_view_event_value = trainset.relevantvalue_view_test

    for step in range(1, training_steps + 1):
        #print(step)
        batch_x, batch_y, batch_seqlen, batch_key_search, batch_key_view, batch_value_search, batch_value_view = trainset.next(batch_size)
        
        if len(batch_x) != batch_size:
            print("Exception!!!")
            continue
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       seqlen: batch_seqlen,relevantevent_search_key: batch_key_search,relevantevent_search_value: batch_value_search,relevantevent_view_key: batch_key_view,relevantevent_view_value: batch_value_view})
        if step % display_step == 0:# or step == 1:
            # Calculate batch accuracy & loss
            #acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,seqlen: batch_seqlen})
            acc, loss = sess.run([accuracy, cost], feed_dict={x: trainset.data_train, y: trainset.labels_train,seqlen: trainset.seqlen_train,relevantevent_search_key: train_relevant_search_event_key,relevantevent_search_value: train_relevant_search_event_value,relevantevent_view_key: train_relevant_view_event_key,relevantevent_view_value: train_relevant_view_event_value})
            print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen,relevantevent_search_key: test_relevant_search_event_key,relevantevent_search_value: test_relevant_search_event_value,relevantevent_view_key: test_relevant_view_event_key,relevantevent_view_value: test_relevant_view_event_value}))
    
    print("Optimization Finished!")

    # Calculate accuracy
    test_data = trainset.data_test
    test_label = trainset.labels_test
    test_seqlen = trainset.seqlen_test
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen,relevantevent_search_key: test_relevant_search_event_key,relevantevent_search_value: test_relevant_search_event_value,relevantevent_view_key: test_relevant_view_event_key,relevantevent_view_value: test_relevant_view_event_value}))
    # Save the variables to disk.
    save_path = saver.save(sess, model_path + "/model.ckpt")
    print("Model saved in file: %s" % save_path)
  
  
    
    