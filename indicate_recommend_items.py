
import re
import numpy as np
import scipy
from scipy import spatial
import os

def cal_simi_twovecs(fun_temp_most_relevant_view_event_words, fun_temp_item):
    #vector_one = np.array(fun_temp_most_relevant_view_event_words)
    vector_one = np.asarray(fun_temp_most_relevant_view_event_words, dtype=float)
    #vector_two = np.array(fun_temp_item)
    vector_two = np.asarray(fun_temp_item, dtype=float)
    temp_distance = spatial.distance.cosine(vector_one, vector_two)
    return temp_distance

search_event_words_list = []
search_event_index_list = []
view_event_words_list = []
view_event_index_list = []

data_train = []
labels_train = []
seqlen_train = []
relevantkey_search_train = []
relevantvalue_search_train = []
relevantkey_view_train = []
relevantvalue_view_train = []

global_index_file = "global_index_file"
if os.path.isfile(global_index_file):
    os.remove(global_index_file)
file_write_global_index_file = open(global_index_file, 'w')

#get userfeature + ad corpus
simi_threshold = 0.9
userid_dic = {}
finish_process_dic = {}
num_different_userid = 0
max_len_sequence = 100
num_features = 16 + 20
vector_zero = []
has_is_list = []

for i_zero_index in range(num_features):
    vector_zero.append(0.0)
#vector_zero = np.array(vector_zero)
#attention_userplusad_file = "tensorflow_sample_100000_newid.csv"
#attention_userplusad_file = "tensorflow_train_sample_100000_20171104_newid.csv"
#attention_userplusad_file = "purchased_end_tensorflow_train_sample_100000_20171130_newid_combined.csv"
attention_userplusad_file = "ebay_linenum_2_title_file_addtocart_end_feature.csv"

trainX = []
trainY = []
current_sequence_list = []
current_feature_list = []
current_num_display_ad = 0
line_num = 0
train_num_point = 80000
haha_num = 0
last_userid = "202"
different_user_num = 0
different_hasyes_user_num = 0
flag_true = False
yes_or_no_list = []

for line_original in open(attention_userplusad_file, 'r'):
    line_num += 1
    if line_num % 100000 == 0:
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
        file_write_global_index_file.write("not" + "\n")
        yes_or_no_list.append("not")
        continue
    if temp_userid not in userid_dic:
        different_user_num += 1
        flag_true = False
        userid_dic[temp_userid] = temp_userid
        haha_num += 1
        num_different_userid += 1
        if current_num_display_ad != 0:
            finish_process_dic[last_userid] = last_userid
            remian_num_display_ad = max_len_sequence - current_num_display_ad
            for i_temp in range(remian_num_display_ad):
                current_sequence_list.append(vector_zero)
            
    
            if len(search_event_words_list) == 0:
                relevantkey_search_train.append(0)
                relevantvalue_search_train.append(0)
            else:
                final_initial_search = search_event_index_list[0]
                temp_most_relevant_search_event_words = search_event_words_list[len(search_event_words_list) - 1]
                for index_list in range(len(search_event_words_list)):
                    if index_list == len(search_event_words_list) - 1:
                        continue
                    temp_item = search_event_words_list[index_list]
                    temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                    if temp_similarity >= simi_threshold and temp_similarity < 0.99:
                        final_initial_search = search_event_index_list[index_list]
                        break

            if len(view_event_words_list) == 0:
                relevantkey_view_train.append(0)
                relevantvalue_view_train.append(0)                        
            else:
                final_initial_view = view_event_index_list[0]
                temp_most_relevant_view_event_words = view_event_words_list[len(view_event_words_list) - 1]
                for index_list in range(len(view_event_words_list)):
                    if index_list == len(view_event_words_list) - 1:
                        continue
                    temp_item = view_event_words_list[index_list]
                    temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                    if temp_similarity >= simi_threshold and temp_similarity < 0.99:
                        final_initial_view = view_event_index_list[index_list]
                        break

            
            #get index of initial and most relevant
            relevant_list = []
            if len(relevant_list) > 0:
                pass
            if len(view_event_words_list) == 0 or final_initial_view == view_event_index_list[0] or final_initial_view == view_event_index_list[len(view_event_words_list) - 2]:
                if len(view_event_words_list) == 0:
                    pass
                else:
                    if final_initial_view == view_event_index_list[0]:
                        relevant_list.append(final_initial_view)
                        relevant_list.append(view_event_index_list[len(view_event_words_list) - 1])
                    if final_initial_view == view_event_index_list[len(view_event_words_list) - 2]:
                        relevant_list.append(final_initial_view)
                        relevant_list.append(view_event_index_list[0])
            elif len(search_event_words_list) == 0 or final_initial_search == search_event_index_list[0] or final_initial_search == search_event_index_list[len(search_event_words_list) - 2]:
                if len(search_event_words_list) == 0:
                    pass
                else:
                    if final_initial_search == search_event_index_list[0]:
                        relevant_list.append(final_initial_search)
                        relevant_list.append(search_event_index_list[len(search_event_index_list) - 1])
                    if final_initial_search == search_event_index_list[len(search_event_words_list) - 2]:
                        relevant_list.append(final_initial_search)
                        relevant_list.append(search_event_index_list[0])
            else:
                relevant_list.append(final_initial_search)
                relevant_list.append(search_event_index_list[len(search_event_index_list) - 1])
            if len(relevant_list) > 0:
                pass   
            for index_temp in range(current_num_display_ad):
                if index_temp in relevant_list:
                    file_write_global_index_file.write("is" + "\n")
                    yes_or_no_list.append("is")
                    has_is_list.append(last_userid)
                    if flag_true == False:
                        different_hasyes_user_num += 1
                        #has_is_list.append(temp_userid)
                        flag_true = True
                
                else:
                    file_write_global_index_file.write("not" + "\n")
                    yes_or_no_list.append("not")
            flag_true = False
            #print(line_num)

            current_sequence_list = []
            current_num_display_ad = 0
            
            
            
            #sucessful store one user, begin another user
            search_event_words_list = []
            search_event_index_list = []
            view_event_words_list = []
            view_event_index_list = []
            current_num_display_ad += 1
            for i in range(num_features):
                current_feature_list.append(float(line_str[i + 1]))
            #current_feature_list = np.array(current_feature_list)
            current_sequence_list.append(current_feature_list)
            current_feature_list = []
            if line_str[3] == '1':
                search_event_words_list.append(line_str[18:38])
                search_event_index_list.append(current_num_display_ad - 1)
            if line_str[1] == '1':
                view_event_words_list.append(line_str[18:38])
                view_event_index_list.append(current_num_display_ad - 1)       
        else:# just for the first line
            current_num_display_ad += 1
            for i in range(num_features):
                current_feature_list.append(float(line_str[i + 1]))
            #current_feature_list = np.array(current_feature_list)
            current_sequence_list.append(current_feature_list)
            current_feature_list = []
            if line_str[3] == '1':
                search_event_words_list.append(line_str[18:38])
                search_event_index_list.append(current_num_display_ad - 1)
            if line_str[1] == '1':
                view_event_words_list.append(line_str[18:38])
                view_event_index_list.append(current_num_display_ad - 1)
    else:
        current_num_display_ad += 1
        for i in range(num_features):
            current_feature_list.append(float(line_str[i + 1]))
        #current_feature_list = np.array(current_feature_list)
        current_sequence_list.append(current_feature_list)
        
        if line_str[3] == '1':
            search_event_words_list.append(line_str[18:38])
            search_event_index_list.append(current_num_display_ad - 1)
        if line_str[1] == '1':
            view_event_words_list.append(line_str[18:38])
            view_event_index_list.append(current_num_display_ad - 1)
        
        if current_num_display_ad == max_len_sequence:
            finish_process_dic[temp_userid] = temp_userid
                #current_sequence_list = np.array(current_sequence_list)

            if len(search_event_words_list) == 0:
                relevantkey_search_train.append(0)
                relevantvalue_search_train.append(0)
            else:
                final_initial_search = search_event_index_list[0]
                temp_most_relevant_search_event_words = search_event_words_list[len(search_event_words_list) - 1]
                for index_list in range(len(search_event_words_list)):
                    if index_list == len(search_event_words_list) - 1:
                        continue
                    temp_item = search_event_words_list[index_list]
                    temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
                    if temp_similarity >= simi_threshold and temp_similarity < 0.99:
                        final_initial_search = search_event_index_list[index_list]
                        break

            if len(view_event_words_list) == 0:
                relevantkey_view_train.append(0)
                relevantvalue_view_train.append(0)                        
            else:
                final_initial_view = view_event_index_list[0]
                temp_most_relevant_view_event_words = view_event_words_list[len(view_event_words_list) - 1]
                for index_list in range(len(view_event_words_list)):
                    if index_list == len(view_event_words_list) - 1:
                        continue
                    temp_item = view_event_words_list[index_list]
                    temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
                    if temp_similarity >= simi_threshold and temp_similarity < 0.99:
                        final_initial_view = view_event_index_list[index_list]
                        break

            relevant_list = []

            if len(view_event_words_list) == 0 or final_initial_view == view_event_index_list[0] or final_initial_view == view_event_index_list[len(view_event_words_list) - 2]:
                if len(view_event_words_list) == 0:
                    pass
                else:
                    if final_initial_view == view_event_index_list[0]:
                        relevant_list.append(final_initial_view)
                        relevant_list.append(view_event_index_list[len(view_event_words_list) - 1])
                    if final_initial_view == view_event_index_list[len(view_event_words_list) - 2]:
                        relevant_list.append(final_initial_view)
                        relevant_list.append(view_event_index_list[0])
            elif len(search_event_words_list) == 0 or final_initial_search == search_event_index_list[0] or final_initial_search == search_event_index_list[len(search_event_words_list) - 2]:
                if len(search_event_words_list) == 0:
                    pass
                else:
                    if final_initial_search == search_event_index_list[0]:
                        relevant_list.append(final_initial_search)
                        relevant_list.append(search_event_index_list[len(search_event_index_list) - 1])
                    if final_initial_search == search_event_index_list[len(search_event_words_list) - 2]:
                        relevant_list.append(final_initial_search)
                        relevant_list.append(search_event_index_list[0])
            else:
                relevant_list.append(final_initial_search)
                relevant_list.append(search_event_index_list[len(search_event_index_list) - 1])
                
            if len(relevant_list) > 0:
                pass 
            for index_temp in range(current_num_display_ad):
                if index_temp in relevant_list:
                    file_write_global_index_file.write("is" + "\n")
                    yes_or_no_list.append("is")
                    has_is_list.append(last_userid)
                    if flag_true == False:
                        different_hasyes_user_num += 1
                        #has_is_list.append(temp_userid)
                        flag_true = True
                else:
                    file_write_global_index_file.write("not" + "\n")
                    yes_or_no_list.append("not")
            
            search_event_words_list = []
            search_event_index_list = []
            view_event_words_list = []
            view_event_index_list = []
            current_sequence_list = []
            current_num_display_ad = 0
            flag_true = False

        current_feature_list = []
        last_userid = temp_userid

    final_initial_search = ""
    final_initial_view = ""
    #process the last userid
if current_num_display_ad != 0:
    remian_num_display_ad = max_len_sequence - current_num_display_ad
    for i_temp in range(remian_num_display_ad):
        current_sequence_list.append(vector_zero)
    
    if line_str[3] == '1':
        search_event_words_list.append(line_str[18:38])
        search_event_index_list.append(current_num_display_ad - 1)
    if line_str[1] == '1':
        view_event_words_list.append(line_str[18:38])
        view_event_index_list.append(current_num_display_ad - 1)
    
    #current_sequence_list = np.array(current_sequence_list)
    seqlen_train.append(current_num_display_ad)
    if len(search_event_words_list) == 0:
        relevantkey_search_train.append(0)
        relevantvalue_search_train.append(0)
    else:
        final_initial_search = search_event_index_list[0]
        temp_most_relevant_search_event_words = search_event_words_list[len(search_event_words_list) - 1]
        for index_list in range(len(search_event_words_list)):
            if index_list == len(search_event_words_list) - 1:
                continue
            temp_item = search_event_words_list[index_list]
            temp_similarity = cal_simi_twovecs(temp_most_relevant_search_event_words, temp_item)
            if temp_similarity >= simi_threshold and temp_similarity < 0.99:
                final_initial_search = search_event_index_list[index_list]
                break

    if len(view_event_words_list) == 0:
        relevantkey_view_train.append(0)
        relevantvalue_view_train.append(0)                        
    else:
        final_initial_view = view_event_index_list[0]
        temp_most_relevant_view_event_words = view_event_words_list[len(view_event_words_list) - 1]
        for index_list in range(len(view_event_words_list)):
            if index_list == len(view_event_words_list) - 1:
                continue
            temp_item = view_event_words_list[index_list]
            temp_similarity = cal_simi_twovecs(temp_most_relevant_view_event_words, temp_item)
            if temp_similarity >= simi_threshold and temp_similarity < 0.99:
                final_initial_view = view_event_index_list[index_list]
                break

    relevant_list = []
    if len(view_event_words_list) == 0 or final_initial_view == view_event_index_list[0] or final_initial_view == view_event_index_list[len(view_event_words_list) - 2]:
        if len(view_event_words_list) == 0:
            pass
        else:
            if final_initial_view == view_event_index_list[0]:
                relevant_list.append(final_initial_view)
                relevant_list.append(view_event_index_list[len(view_event_words_list) - 1])
            if final_initial_view == view_event_index_list[len(view_event_words_list) - 2]:
                relevant_list.append(final_initial_view)
                relevant_list.append(view_event_index_list[0])
    elif len(search_event_words_list) == 0 or final_initial_search == search_event_index_list[0] or final_initial_search == search_event_index_list[len(search_event_words_list) - 2]:
        if len(search_event_words_list) == 0:
            pass
        else:
            if final_initial_search == search_event_index_list[0]:
                relevant_list.append(final_initial_search)
                relevant_list.append(search_event_index_list[len(search_event_index_list) - 1])
            if final_initial_search == search_event_index_list[len(search_event_words_list) - 2]:
                relevant_list.append(final_initial_search)
                relevant_list.append(search_event_index_list[0])
    else:
        relevant_list.append(final_initial_search)
        relevant_list.append(search_event_index_list[len(search_event_index_list) - 1])
    if len(relevant_list) > 0:
        pass 
    for index_temp in range(current_num_display_ad):
        if index_temp in relevant_list:
            file_write_global_index_file.write("is" + "\n")
            yes_or_no_list.append("is")
            has_is_list.append(last_userid)
            if flag_true == False:
                different_hasyes_user_num += 1
                flag_true = True
        else:
            file_write_global_index_file.write("not" + "\n")
            yes_or_no_list.append("not")
    
    data_train.append(current_sequence_list)
    if line_str[num_features + 2] == '1':
        labels_train.append([1.0, 0.0])
    else:
        labels_train.append([0.0, 1.0])
    current_sequence_list = []
    current_num_display_ad = 0
                
print(different_user_num)             
print(different_hasyes_user_num)

new_file = "threshold_" + str(simi_threshold) + "_contain_all_events.csv"
if os.path.isfile(new_file):
    os.remove(new_file)
file_write_new_file = open(new_file, 'w')

attention_userplusad_file = "ebay_linenum_2_title_file_addtocart_end"
temp_line_num = -1
for line_new in open(attention_userplusad_file, 'r'):
    temp_line_num += 1
    if temp_line_num > 2000:
        break
    one_line_new_str = line_new.split("|||")
    temp_user_id = one_line_new_str[0]
    if temp_user_id in has_is_list:
        file_write_new_file.write(line_new.strip() + "\n")
  
new_rec_file = "threshold_" + str(simi_threshold) + "_contain_just_recommend_events.csv"
if os.path.isfile(new_rec_file):
    os.remove(new_rec_file)
file_write_new_rec_file = open(new_rec_file, 'w')  
current_line_num = -1
for next_line_new in open(attention_userplusad_file, 'r'):
    current_line_num += 1
    if current_line_num > 2000:
        break
    one_line_new_str = next_line_new.split("|||")
    two_line_new_str = one_line_new_str[1].split(" ")
    temp_user_id = one_line_new_str[0]
    if temp_user_id not in has_is_list:
        continue
    if yes_or_no_list[current_line_num] == "is":
        file_write_new_rec_file.write(next_line_new.strip() + "\n")
        continue
    if yes_or_no_list[current_line_num] == "not" and two_line_new_str[0] != "SEARCH" and two_line_new_str[0] != "VI":
        file_write_new_rec_file.write(next_line_new.strip() + "\n")
        continue   
    #temp_user_id = one_line_new_str[0]

    
            