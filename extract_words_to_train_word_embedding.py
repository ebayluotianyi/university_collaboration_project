'''
Created on Nov 4, 2017

@author: root
'''

import re
import os

if __name__ == '__main__':
    original_data_file = "ebay_event_file_20170816_100000.csv"
    ebay_linenum_2_title_file = "purchased_end_ebay_words_to_train_word2vector"
    word2id = {}
    word2fre = {}
    line_num = 0
    possible_value_num = 1
    word_num = 0
    dic_page_type = {}
    if os.path.isfile(ebay_linenum_2_title_file):
        os.remove(ebay_linenum_2_title_file)
    file_write_ebay_word2fre_file = open(ebay_linenum_2_title_file, 'w')
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
            temp_result_str = ""
            page_type = ""
            for current_feature in new_str_features:
                str_current_feature = current_feature.split(":")
                name_feature = str_current_feature[0].replace("\"", "")
                #print name_feature
                if name_feature == "page_type":
                    simplified_str_current_feature = str_current_feature[1].replace("\"", "")
                    #print simplified_str_current_feature
                    page_type = simplified_str_current_feature
                    
                if name_feature == "item_title" and page_type == "VI":
                    simplified_str_current_feature = str_current_feature[1].replace("\"", "").replace("+"," ")
                    simplified_str_current_feature = re.sub(r'\s+', ' ', simplified_str_current_feature)
                    current_str = ""
                    if len(simplified_str_current_feature) != 0:
                        #if page_type == "PURCHASE":
                        #   pass
                        #else:
                        file_write_ebay_word2fre_file.write(simplified_str_current_feature.lower() + "\n")
                
                if name_feature == "keyword" and page_type == "SEARCH":
                    has_keyword_or_title = 1
                    simplified_str_current_feature = str_current_feature[1].replace("\"", "")
                    current_str = ""
                    if len(simplified_str_current_feature) != 0:
                        #if page_type == "PURCHASE":
                        #    pass
                        #else:
                        file_write_ebay_word2fre_file.write(simplified_str_current_feature.lower() + "\n")

            if page_type == "PURCHASE":
                break