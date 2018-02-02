'''
Created on 25 Jan, 201

@author: luotianyi
'''
import os
import re

if __name__ == '__main__':
    original_data_file = "ebay_event_file_20170816_100000.csv"
    dic_page_type = {'BIN': 8, 'PURCHASE': 11, 'SEARCH': 3, 'SELL': 12, 'BROWSE': 6, 'VI': 1, 'BID': 10, 'WATCH': 5, 'PRP': 4, 'ADD_TO_CART': 7, 'CART_CHECKOUT': 9, 'HOMEPAGE': 2}

    ebay_data_tetrad_format = "tetrad_sample_1000000_getting_potential_causal_relations_40.csv"
    if os.path.isfile(ebay_data_tetrad_format):
        os.remove(ebay_data_tetrad_format)
    file_write_final_result = open(ebay_data_tetrad_format, 'w')
    line_num = 0
    temp_line_result_str = ""
    max_len_temp = 40
    num_purchase = 0
    temp_result_str = ""
    for j in range(max_len_temp + 1):
        if j == max_len_temp:
            temp_result_str += "purchased_label"
        else:
            temp_result_str += "sequence_" + str(j + 1) + " "
    file_write_final_result.write(temp_result_str + "\n")
    
    for line in open(original_data_file, 'r'):
        #print(line)
        line_num += 1
        if "PURCHASE" in line:
            label = "1.0"
            num_purchase += 1
        else:
            label = "0.0"
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
            if max_len_temp < page_num:
                break
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
                            label = "1.0"
                            #num_purchase += 1
                        if max_len_temp == page_num:
                            temp_line_result_str += str(float(current_page_type_id)) + " " + label
                            break
                        else:
                            temp_line_result_str += str(float(current_page_type_id)) + " "
        if len(temp_line_result_str) > 0:                   
            if page_num < max_len_temp:
                remain_num = max_len_temp - page_num
                for i in range(remain_num):
                    if i == remain_num - 1:
                        temp_line_result_str += "0.0 " + label
                    else:
                        temp_line_result_str += "0.0 "
            
            file_write_final_result.write(temp_line_result_str + "\n")
        temp_line_result_str = ""
    print("All finish!!!")
    print(num_purchase)
    
    