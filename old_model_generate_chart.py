# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import random
import numpy as np
import re

#26,259,078
#26258235
#train: 21,000,000
#test: 5,259,078

# ==========
#   MODEL
# ==========

# Parameters
learning_rate = 0.01
training_steps = 40000
batch_size = 100
display_step = 1000

# Network Parameters
seq_max_len = 20 # Sequence max length
n_hidden = 64 # hidden layer num of features
n_classes = 2 # linear sequence or not

#trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
#testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# tf Graph input
x = tf.placeholder("float", [None, seq_max_len, 36])
y = tf.placeholder("float", [None, n_classes])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

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

pred = dynamicRNN(x, seqlen, weights, biases)
pred_softmax = tf.nn.softmax(logits=pred)
# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
model_path = "/home/shaotang/Desktop/eclipseworkspace/deep_learning_causal_models_tensorflow/20171203_old_model"

# Start training
with tf.Session() as sess:
    # Run the initializer and load the model
    sess.run(init)
    saver.restore(sess, model_path + "/model.ckpt")

    #get the test data
    current_id = "47048"
    max_len = 20
    num_features = 16 + 20
    # Network building
    dic_page_type = {'BIN': 8, 'PURCHASE': 11, 'SEARCH': 3, 'SELL': 12, 'BROWSE': 6, 'VI': 1, 'BID': 10, 'WATCH': 5, 'PRP': 4, 'ADD_TO_CART': 7, 'CART_CHECKOUT': 9, 'HOMEPAGE': 2}
    page_type2num = {}
    for temp_item in dic_page_type:
        page_type2num[dic_page_type[temp_item]] = temp_item
    dic_device = {'Mobile': 1, 'PC': 2, 'Other': 3}
    device2num = {}
    for temp_item in dic_device:
        device2num[dic_device[temp_item]] = temp_item
    #original_data_file = "tensorflow_sample_100000_newid_purchased.csv"
    #original_data_file = "tensorflow_train_sample_100000_20171104_newid.csv"
    original_data_file = "purchased_end_tensorflow_train_sample_100000_20171130_newid.csv"
    #original_data_file = "newcorpus_all_purchased_end_tensorflow_train_sample_100000_20171113_newid.csv"
    temp_current_sequence_list = []
    temp_current_sequence_list_top = []
    vector_zero = []
    num = 0
    for i_zero_index in range(num_features):
        vector_zero.append(0.0)
    current_feature_list = []
    current_sequence_list = []
    temp_current_sequence_list_top = []
    new_current_sequence_list = []
    purchase_prob_decrase = []
    log_file = "old_model_predict_file_" + current_id
    log_writer = open(log_file, 'w')
    line_num = 0
    current_num = 0
    for line in open(original_data_file, 'r'):
        line_num += 1
        if line_num % 5000000 == 0:
            print(line_num)
        temp_purchase_prob_decrase = []
        line_str = line.split(" ")
        temp_userid = line_str[0]
        if line_str[0] == current_id:
            line_simplify = re.sub(r'\s+', ' ', line.strip())                
            line_str = line_simplify.split(" ")
            current_num += 1
            #if current_num <= 18:
            #    continue
            if len(line_str) > 10:
                if len(line_str) == 19:
                    for i in range(num_features - 16):
                        line_str.append("0.0")
                    temp_label = line_str[16 + 2]
                    line_str[16 + 2] = line_str[num_features + 2]
                    line_str[num_features + 2] = temp_label
            else:
                continue
            #print(len(line_str))
            print(current_num)
            for i in range(num_features):
                current_feature_list.append(float(line_str[i + 1]))
            current_feature_list = np.array(current_feature_list)
            current_sequence_list.append(current_feature_list)
            for temp_item in current_sequence_list:
                new_current_sequence_list.append(temp_item)
            num += 1
            remian_num_display_ad = max_len - num
            if remian_num_display_ad == -1:
                break
            for i_temp in range(remian_num_display_ad):
                new_current_sequence_list.append(vector_zero)
            temp_current_sequence_list_top.append(new_current_sequence_list)
            #temp_pro = model.predict(temp_current_sequence_list_top)
            
            test_seqlen_temp = []
            test_data = temp_current_sequence_list_top
            test_seqlen_temp.append(num)
            #print(sess.run(pred_softmax, feed_dict={x: test_data, seqlen: test_seqlen_temp}))
            log_writer.write(str(current_num) + "\n")
            print(sess.run(pred_softmax, feed_dict={x: test_data, seqlen: test_seqlen_temp}))
            log_writer.write(str(sess.run(pred_softmax, feed_dict={x: test_data, seqlen: test_seqlen_temp})) + "\n")
            for i in range(num_features):
                if line_str[i + 1] == "1":
                    if i < 13:
                        print("page_type")
                        log_writer.write("page_type" + "\n")
                        print(page_type2num[i + 1])
                        log_writer.write(str(page_type2num[i + 1]) + "\n")
                    elif i < 16:
                        print("device")
                        log_writer.write("device" + "\n")
                        print(device2num[i - 13 + 1])
                        log_writer.write(str(device2num[i - 13 + 1]) + "\n")
            print("\n")
            log_writer.write("\n")
        
        current_feature_list = []
        new_current_sequence_list = []
        temp_current_sequence_list_top = []
    
    
    
     