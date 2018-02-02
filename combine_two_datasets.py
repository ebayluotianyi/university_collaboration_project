import os

data_file_one = "purchased_end_tensorflow_train_sample_100000_20171130_newid_for_combine.csv"
data_file_two = "newcorpus_all_purchased_end_tensorflow_train_sample_100000_20171203_newid.csv"
data_file = "purchased_end_tensorflow_train_sample_100000_20171130_newid_combined.csv"
if os.path.isfile(data_file):
    os.remove(data_file)
line_num = 0
file_write_final = open(data_file, 'w')
    
for line_one in open(data_file_one, 'r'):
    line_num += 1
    line_str = line_one.split(" ")
    if int(line_str[0]) >= 80000:
        break
    if line_num % 500000 == 0:
        print(line_num)
    new_line_one = line_one.strip()
    file_write_final.write(new_line_one + "\n")
    
for line_two in open(data_file_two, 'r'):
    line_num += 1
    line_str = line_two.split(" ")
    if int(line_str[0]) >= 80000:
        break
    if line_num % 500000 == 0:
        print(line_num)
    new_line_two = line_two.strip()
    file_write_final.write(new_line_two + "\n")
    
for line_one in open(data_file_one, 'r'):
    line_num += 1
    line_str = line_one.split(" ")
    if int(line_str[0]) < 80000:
        break
    if line_num % 500000 == 0:
        print(line_num)
    new_line_one = line_one.strip()
    file_write_final.write(new_line_one + "\n")
    
for line_two in open(data_file_two, 'r'):
    line_num += 1
    line_str = line_two.split(" ")
    if int(line_str[0]) < 80000:
        break
    if line_num % 500000 == 0:
        print(line_num)
    new_line_two = line_two.strip()
    file_write_final.write(new_line_two + "\n")
    
file_write_final.close()
