'''
Created on Nov 2, 2017

@author: root
'''

if __name__ == '__main__':
    show_line_num = "47048"
    linenum_title_words_file = "ebay_linenum_2_title_file_purchased_end"
    current_title_words_file = "current_title_words_file"
    write_current_title_words = open(current_title_words_file, "w")
    #generate_file = ""
    num = 0
    for line in open(linenum_title_words_file, "r"):
        line_str = line.strip().split("|||")
        if line_str[0] == show_line_num:
            num += 1
            print(str(num) + "\n" + line + "\n")
            write_current_title_words.write(str(num) + "\n" + line + "\n")
    print(num)
        