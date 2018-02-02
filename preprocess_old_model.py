'''
Created on Aug 16, 2017

@author: tialuo
'''
import re
import os
import random
import numpy as np

if __name__ == '__main__':
    #get word embedding dictionary
    file_ebay_word_emb = "purchased_end_ebay_100000_20171130_vectors_nobin"
    dic_word2vec = {}
    emb_num = 0
    for line in open(file_ebay_word_emb, 'r'):
        emb_num += 1
        #if emb_num > 50000:
        #    break
        if emb_num % 50000 == 0:
            print(emb_num)
        line_str = line.split(" ")
        if len(line_str) == 22:
            for i in range(20):
                line_str[i + 1] = float(line_str[i + 1])
            embe_str = line_str[1:len(line_str) - 1]
            #print len(embe_str)
            dic_word2vec[line_str[0]] = np.array(embe_str)
            #print line
    print("The vocabulary size is " + str(len(dic_word2vec)))
    
    #trainingset_index_list = random.sample(range(0, 40000), 32000)
    trainingset_index_list = list(range(0, 100000))
    trainingset_index_dic = {}
    for temp_index in trainingset_index_list:
        trainingset_index_dic[temp_index] = temp_index
    ebay_data_tensorflow_format_train = "purchased_end_tensorflow_train_sample_100000_20171130_newid.csv"
    ebay_data_tensorflow_format_test = "tensorflow_test_sample_100000_20171104_newid.csv"
    if os.path.isfile(ebay_data_tensorflow_format_train): 
        os.remove(ebay_data_tensorflow_format_train)
    if os.path.isfile(ebay_data_tensorflow_format_test): 
        os.remove(ebay_data_tensorflow_format_test)
    file_write_final_result_train = open(ebay_data_tensorflow_format_train, 'w')
    file_write_final_result_test = open(ebay_data_tensorflow_format_test, 'w')
    
    original_data_file = "ebay_event_file_20170816_100000.csv"
    dic_page_type = {'BIN': 8, 'PURCHASE': 11, 'SEARCH': 3, 'SELL': 12, 'BROWSE': 6, 'VI': 1, 'BID': 10, 'WATCH': 5, 'PRP': 4, 'ADD_TO_CART': 7, 'CART_CHECKOUT': 9, 'HOMEPAGE': 2}
    dic_device = {'Mobile': 1, 'PC': 2, 'Other': 3}
    dic_word2id = {}
    ebay_event_onehot_feature_file = "ebay_event_onehot_feature"
    if os.path.isfile(ebay_event_onehot_feature_file): 
        os.remove(ebay_event_onehot_feature_file)
    file_write_ebay_event_onehot_feature_file = open(ebay_event_onehot_feature_file, 'w')
    for word2id_line in open("ebay_word2id_file", 'r'):
        str_word2id = word2id_line.strip().split(" ")
        dic_word2id[str_word2id[0]] = int(str_word2id[1])
    line_num = 0
    num_purchase = 0
    num_vocabulary = 300
    label = "0"
    no_words = False
    for line in open(original_data_file, 'r'):
        if "PURCHASE" in line:
            label = "1"
            num_purchase +=1
        else:
            label = "0"
        line_num += 1
        #if line_num >= 40000:
        #    break
        #print line_num
        if line_num % 1000 == 0:
            print(line_num)
        single_space_line = re.sub(' +', ' ', line)
        str_single_space_line = single_space_line.strip().split("\t")
        str_pages = str_single_space_line[7].split("},{")
        pages_len = len(str_pages)
        page_num = 0
        for current_page in str_pages:
            embe_str = ""
            temp_str = ""
            page_num += 1
            if page_num == 1 or page_num == pages_len:
                if page_num == 1:
                    new_str = current_page[2:len(current_page)]
                else:
                    new_str = current_page[:len(current_page)-2]
            else:
                new_str = current_page
            new_str_features = new_str.split(",")
            page_type = ""
            for current_feature in new_str_features:
                str_current_feature = current_feature.split(":")
                name_feature = str_current_feature[0].replace("\"", "")
                if name_feature == "page_type":
                    simplified_str_current_feature = str_current_feature[1].replace("\"", "")
                    page_type = simplified_str_current_feature
                    if len(simplified_str_current_feature) !=0:
                        current_page_type_id = dic_page_type[simplified_str_current_feature]
                        if current_page_type_id == 11:
                            label = "1"
                            num_purchase +=1
                    else:
                        current_page_type_id = 13
                    for i_page_type in range(13):
                        if i_page_type == current_page_type_id - 1:
                            temp_str += "1 "
                        else:
                            temp_str += "0 "
                if name_feature == "device":
                    simplified_str_current_feature = str_current_feature[1].replace("\"", "")
                    if len(simplified_str_current_feature) !=0:
                        current_page_type_id = dic_device[simplified_str_current_feature]
                    else:
                        current_page_type_id = 4
                    for i_page_type in range(4):
                        if i_page_type == current_page_type_id - 1:
                            temp_str += "1 "
                        else:
                            temp_str += "0 "
                if name_feature == "item_title" and page_type == "VI":
                    simplified_str_current_feature = str_current_feature[1].strip().replace("\"", "").replace("+"," ")
                    simplified_str_current_feature = re.sub(r'\s+', ' ', simplified_str_current_feature).lower()
                    if len(simplified_str_current_feature) !=0:
                        word_list = simplified_str_current_feature.split(" ")
                        current_word_list = []
                        embe_str = ""
                        for temp_word in word_list:
                            if temp_word in dic_word2vec:
                                #print "in"
                                current_embe = dic_word2vec[temp_word]
                                current_word_list.append(current_embe)
                            else:
                                #print "not in"
                                pass
                        current_word_array = np.array(current_word_list) 
                        ave_current_word_array = np.mean(current_word_array, axis=0)
                        #print ave_current_word_array 
                        if len(str(ave_current_word_array)) > 10:
                            for i in range(20):
                                if i != 19:
                                    embe_str += str(ave_current_word_array[i]) + " "
                                else:
                                    embe_str += str(ave_current_word_array[i])
                        else:
                            for i in range(20):
                                if i != 19:
                                    embe_str += "0.0" + " "
                                else:
                                    embe_str += "0.0"
                    else:
                        no_words = True
                        for i in range(20):
                            if i != 19:
                                embe_str += "0.0" + " "
                            else:
                                embe_str += "0.0"
                if name_feature == "keyword" and page_type == "SEARCH":
                    simplified_str_current_feature = str_current_feature[1].strip().replace("\"", "").replace("+"," ")
                    simplified_str_current_feature = re.sub(r'\s+', ' ', simplified_str_current_feature).lower()
                    if len(simplified_str_current_feature) !=0:
                        word_list = simplified_str_current_feature.split(" ")
                        current_word_list = []
                        embe_str = ""
                        for temp_word in word_list:
                            if temp_word in dic_word2vec:
                                #print "in"
                                current_embe = dic_word2vec[temp_word]
                                current_word_list.append(current_embe)
                            else:
                                #print "not in"
                                pass
                        current_word_array = np.array(current_word_list) 
                        ave_current_word_array = np.mean(current_word_array, axis=0)
                        #print ave_current_word_array 
                        if len(str(ave_current_word_array)) > 10:
                            for i in range(20):
                                if i != 19:
                                    embe_str += str(ave_current_word_array[i]) + " "
                                else:
                                    embe_str += str(ave_current_word_array[i])
                        else:
                            for i in range(20):
                                if i != 19:
                                    embe_str += "0.0" + " "
                                else:
                                    embe_str += "0.0"
                    else:
                        no_words = True
                        for i in range(20):
                            if i != 19:
                                embe_str += "0.0" + " "
                            else:
                                embe_str += "0.0"
            #file_write_ebay_event_onehot_feature_file.write(temp_str + " " + label + "\n")
            if label == "1":
                if no_words == False:
                #if page_type == "PURCHASE":
                #    pass
                #else:
                    file_write_final_result_train.write(str(line_num) + " " + temp_str + embe_str + " " + label + "\n")
            else:
                #if page_type == "PURCHASE":
                #    pass
                #else:
                if no_words == False:
                    file_write_final_result_train.write(str(line_num) + " " + temp_str + embe_str + " " + label + "\n")    
            no_words = False
            if page_type == "PURCHASE":
                break
    print(num_purchase)