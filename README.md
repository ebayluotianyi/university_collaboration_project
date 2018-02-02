# university_collaboration_project
1.extract_words_to_train_word_embedding.py: get the word line by line to train the word2vec
<p>
input:
eBay_event_file_20170816_10000.csv #8.6GB (too big to upload). It contains 100,000 sequences of users behaviors events.
<p>
output:
purchased_end_ebay_words_to_train_word2vector

2.train the word embedding with mikolov word2vec's c code:
<p>
input: purchased_end_ebay_words_to_train_word2vector (too big to upload)
<p>
output: purchased_end_ebay_100000_20171130_vectors_nobin (too big to upload)

3.preprocess_old_model.py: preprocess the corpus and generate the feature file
<p>
input:
purchased_end_ebay_100000_20171130_vectors_nobin (too big to upload)
(Use generate_words_dic.py to generate “ebay_word2id_file")
<p>
output: purchased_end_tensorflow_train_sample_100000_20171130_newid.csv (too big to upload)

4.lstm_old_model_tensorflow.py: graph lstm model training
<p>
input: 
purchased_end_tensorflow_train_sample_100000_20171130_newid.csv (too big to upload)
<p>
output:
model.ckpt
