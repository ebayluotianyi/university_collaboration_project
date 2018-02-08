# university_collaboration_project
I modified the sourcecode of chain LSTM model in the tensorflow and developed a new graph-lstm model.

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

4.lstm_old_model_tensorflow.py: graph lstm model training and testing
<p>
input: 
purchased_end_tensorflow_train_sample_100000_20171130_newid.csv (too big to upload)
<p>
output:
model.ckpt
  
5. indicate_recommend_items.py: Filter irrelevant search or view events and leave the relevant ones to make users purchase more things.
"threshold_0.9_contain_all_events.csv" is the file which contains the sequences of users behavior events including "add_to_cart" event.
<p>
The format of "threshold_0.9_contain_all_events.csv" is “[userid]|||[page type] [Device] [title words for view page or searched keywords for search papge]” (One example: "449|||SEARCH Mobile tokyo disney". It means user 449 search the "tokyo disney" in his or her mobile device.).
<p>
"threshold_0.9_contain_just_recommend_events.csv" is the file which contains events which filter the irrelevant search or view events. The format is the same as the "threshold_0.9_contain_all_events.csv".
