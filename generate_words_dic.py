'''
Created on Aug 17, 2017

@author: tialuo
'''
import re
import os

if __name__ == '__main__':
    original_data_file = "ebay_event_file_20170816_100000.csv"
    ebay_word2fre_file = "ebay_word2fre_file"
    ebay_word2id_file = "ebay_word2id_file"
    word2id = {}
    word2fre = {}
    line_num = 0
    possible_value_num = 1
    word_num = 0
    dic_page_type = {}
    if os.path.isfile(ebay_word2fre_file): 
        os.remove(ebay_word2fre_file)
    if os.path.isfile(ebay_word2id_file):
        os.remove(ebay_word2id_file)
    file_write_ebay_word2fre_file = open(ebay_word2fre_file, 'w')
    file_write_ebay_word2id_file = open(ebay_word2id_file, 'w')
    #2645374
    for line in open(original_data_file, 'r'):
        line_num += 1
        #if line_num > 1000:
        #    break
        if line_num % 10000 == 0:
            print(line_num)
        single_space_line = re.sub(' +', ' ', line)
        str_single_space_line = single_space_line.strip().split("\t")
        str_pages = str_single_space_line[7].split("},{")
        pages_len = len(str_pages)
        page_num = 0
        for current_page in str_pages:
            page_num += 1
            if page_num == 1 or page_num == pages_len:
                if page_num == 1:
                    new_str = current_page[2:len(current_page)]
                else:
                    new_str = current_page[:len(current_page)-2]
            else:
                new_str = current_page
            new_str_features = new_str.split(",")
            for current_feature in new_str_features:
                str_current_feature = current_feature.split(":")
                name_feature = str_current_feature[0].replace("\"", "")
                #if name_feature == "page_type":
                #    print("haha")
                    #dic_page_type[] = 
                if name_feature == "item_title":
                    simplified_str_current_feature = str_current_feature[1].replace("\"", "")
                    current_str = ""
                    if len(simplified_str_current_feature) !=0:
                        word_list = simplified_str_current_feature.split("+")
                        for temp_word in word_list:
                            if len(temp_word) != 0:
                                temp_word_lower = temp_word.lower()
                                if temp_word_lower not in word2fre:
                                    word2fre[temp_word_lower] = 1
                                    word_num += 1
                                else:
                                    word2fre[temp_word_lower] += 1
    final_dic= sorted(word2fre.items(), key=lambda d:d[1], reverse = True)     
    num_final_dic = -1
    for temp_word in final_dic:
        num_final_dic += 1
        if num_final_dic >= 300:
            break
        file_write_ebay_word2fre_file.write(temp_word[0] + " " + str(temp_word[1]) + "\n")
        file_write_ebay_word2id_file.write(temp_word[0] + " " + str(num_final_dic + 1) + "\n")