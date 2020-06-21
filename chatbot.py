# Building a chatbot with Deep NLP

#importing the librairies

import numpy as np
import tensorflow as tf
import re
import time
import pandas as pd
import csv




################  PART 1 - Data preprocessing  ################

# df = dataframe, helps with visualisation
df = pd.read_csv("counsel_chat.csv", encoding='utf-8')

# put the csv file into a list
with open('counsel_chat.csv', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    csvData = list(reader)

# creating two seperate lists for questions and answers
questions = []
answers = []

#skip the first row (because it contains the names of the columns)
iterCsvData = iter(csvData)
next(iterCsvData)

for ligne in iterCsvData:
    questions.append(ligne[3])
    answers.append(ligne[8])

# making a first cleaning of the text
def clean_text(text):
    # put all the text into lower cases
    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"don't", "do not", text)
    # removing the punctuation
    text = re.sub(r"[-()\"#@/;:<>{}+=?~|.,]", "", text)
    return text
                 
# cleaning the questions (by applying the function)
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
# cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
# removing the non frequent words
## creating a dictionary that maps each word to its number of occurences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# creating two dictionaries that maps the questions words and the answers to a unique integer
threshold = 5
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1
        
# adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int)+1

for token in tokens:
    answerswords2int[token] = len(answerswords2int)+1
    
# creating the inverse dictionary of the answerswords2int dictionary (useful for the seq2seq model)
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

# adding the EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# translating all the questions and answers into integers
# and replacing all the words that were filtered out by <OUT>
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

#sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 30+1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
            
            
            
            
################  PART 2 - Building the Seq2Seq model  ################

# creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    # hold the learning rate
    lr = tf.placeholder(tf.float32, name='learning_rate')
    # hold the keep probability parameter ( used to control the drop out rate )
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob

# preprocessing the targets ( because the decoder only accept a certain format of targets:
#           - target must be into batches 
#           - each target of the batch must start with SOS token )
def preprocess_targets(targets, word2int, batch_size):
    # get the left side of the concatenation ( vector of batch size elements only containing the SOS tokens )
    left_side = tf.fill([batch_size, 1], word2int ['<SOS>'])
     # get the right side of the concatenation ( all the batch size answers in the batch except the last column that conatains EOS tokens won't be needed )
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_target = tf.concat([left_side, right_side], 1)
    return preprocessed_target

# creating the Encoder RNN Layer
    # rnn_inputs = the model inputs
    # rnn_size = number of input tensors of the encoder layer we are making
    # keep_prob = apply a drop out regularization to the lstm cell ( desactivating a certain % of the neurons during the training )
    # sequence_length = list containing the length of each question in the batch 
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

# decoding the training set
    ## encoder state = cause the decoder is getting the encoder state as part of the input to proceed the decoding
    ## decoder_cell = cell in the RNN decoder
    ## decoder_embedding_input = inputs in which we applied embedding
    ## decoding_scope = will wrap the TF variable
    ## output_function = function we will use to return the decoder output in the end
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    # preprocess the training data in order to prepare it to the attention process
      ## attention_keys = keys that will be compared to the attention state
      ## attention_values = values that will be used to construct the context vector
      ## attention_score_function = used to compute the similarity between the keys and the target states
      ## attention_construct_function = used to build the attention states
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    # attentional decoding function that will decode the training set
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    # get the final output of the decoder
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                  training_decoder_function,
                                                                  decoder_embedded_input,
                                                                  sequence_length,
                                                                  scope = decoding_scope)
    # apply a final dropout to the decoder output
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                    test_decoder_function,
                                                                    scope = decoding_scope)
    return test_predictions

# creating the decoder RNN
    ## num_layers = number of layers we want to have inside the RNN of our decoder
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        # initialize the weights that will be associated to the neurons of the fully connected layers of the decoder
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        # initialize the biases as zeros
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell, 
                                           decoder_embeddings_matrix, 
                                           word2int['<SOS>'],
                                           word2int['<EOS>'], 
                                           sequence_length -1, 
                                           num_words, 
                                           decoding_scope, 
                                           output_function, 
                                           keep_prob, 
                                           batch_size)
    return training_predictions, test_predictions

# building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    ## get the encoder state (what is returned by the encoder) that will be the input of the decoder
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    ## get the preprocessed targets (needed for the training) and the decoder embeddings matrix (to get the decoder embeddings input)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions




################  PART 3 - Training the Seq2Seq model  ################















