# university_collaboration_project
Old model:
1.extract_words_to_train_word_embedding.py: get the word line by line to train the word2vec
input:
eBay_event_file_20170816_10000.csv#8.6GB(too big to upload). It contains 100,000 sequences of users behaviors events.
output:
purchased_end_ebay_words_to_train_word2vector

2.train the word embedding with mikolov word2vec's c code:
input: purchased_end_ebay_words_to_train_word2vector
output: purchased_end_ebay_100000_20171130_vectors_nobin

3.preprocess_old_model.py: preprocess the corpus and generate the feature file
input:
purchased_end_ebay_100000_20171130_vectors_nobin
(Use generate_words_dic.py to generate “ebay_word2id_file")
output: purchased_end_tensorflow_train_sample_100000_20171130_newid.csv

4.lstm_old_model_tensorflow.py: graph lstm model training
input: 
purchased_end_tensorflow_train_sample_100000_20171130_newid.csv
output:
model.ckpt
