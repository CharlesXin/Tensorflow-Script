#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-09-10
# @Author  : Ivan
import os, sys
os.chdir("C:\\Users\\cxin\\Desktop\\计算机视觉资料\\Lecture6_0910\\rnn_sentiment")
import numpy as np
import logging
import time
import random
import argparse
import tensorflow as tf
from wordvec import WordEmbedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, GlobalAveragePooling1D
from keras.utils import to_categorical


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', help='The size of each batch used to be trained.', type=int, default=100)
parser.add_argument('-l', '--rate', help='Learning rate of AdaGrad.', type=float, default=0.02)
parser.add_argument('-e', '--epoch', help='Number of the training epoch.', type=int, default=25)
parser.add_argument('-m', '--model', help="Which model to use", type=str, default="rnn")
parser.add_argument('-n', '--name', help='Name used to save the model.', type=str, default="sentiment")
parser.add_argument('-d', '--seed', help='Random seed used for generation.', type=int, default=42)
# Compile and configure all the parameters.
args = parser.parse_args()
np.random.seed(args.seed)
tf.set_random_seed(args.seed)
# Logging configuration.
# Set the basic configuration of the logging system.
log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s', 
                                  datefmt='%m-%d %H:%M')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# File logger
file_handler = logging.FileHandler("{}.log".format("sentiment"))
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
# Stderr logger
std_handler = logging.StreamHandler(sys.stdout)
std_handler.setFormatter(log_formatter)
std_handler.setLevel(logging.DEBUG)
logger.addHandler(std_handler)
# This script is used to train and test on the Movie-Review dataset
# Since this data set is not explicitly partitioned into training/test 
# split, we use 0.7/0.3 partition to as the train/test split.
mr_positive_filename = './mr-polarity.pos'
mr_negative_filename = './mr-polarity.neg'
# Loading and building training and test data set.
mr_txt, mr_label = [], []
start_time = time.time()
# Read all the instances 
with open(mr_positive_filename, 'r') as fin:
    lines = fin.readlines()
    logger.info('Positive Instances: %d' % len(lines))
    mr_txt.extend(lines)
    mr_label.extend([1] * len(lines))
with open(mr_negative_filename, 'r') as fin:
    lines = fin.readlines()
    logger.info('Negative Instances: %d' % len(lines))
    mr_txt.extend(lines)
    mr_label.extend([0] * len(lines))
# Shuffling all the data.
assert len(mr_txt) == len(mr_label)
data_size = len(mr_txt)
logger.info('Size of the data sets: %d' % data_size)
random_index = np.arange(data_size)
np.random.shuffle(random_index)
mr_txt = list(np.asarray(mr_txt)[random_index])
mr_label = list(np.asarray(mr_label)[random_index])
end_time = time.time()
# Record timing
logger.info('Time used to load and shuffle MR dataset: %f seconds.' % (end_time-start_time))
# Load word-embedding
embedding_filename = './wiki_embeddings.txt'
# Load training/test data sets and wiki-embeddings.
word_embedding = WordEmbedding(embedding_filename)
embed_dim = word_embedding.embedding_dim()
start_time = time.time()
blank_index = word_embedding.word2index('</s>')
logger.info('Blank index: {}'.format(word_embedding.index2word(blank_index)))
# Word-vector representation, zero-padding all the sentences to the maximum length.
max_len = 52
mr_insts = np.zeros((data_size, max_len, word_embedding.embedding_dim()), dtype=np.float32)
mr_label = np.asarray(mr_label)[:, np.newaxis]
for i, sent in enumerate(mr_txt):
    words = sent.split()
    words = [word.lower() for word in words]
    l = len(words)
    # vectors = np.zeros((len(words)+2, embed_dim), dtype=np.float32)
    mr_insts[i, 1: l+1, :] = np.asarray([word_embedding.wordvec(word) for word in words])
    mr_insts[i, 0, :] = mr_insts[i, l+1, :] = word_embedding.wordvec("</s>")
end_time = time.time()
logger.info('Time used to build sparse and dense input word-embedding matrices: %f seconds.' % (end_time-start_time))
logger.info("Shape of data tensor = {}, shape of label matrix = {}".format(mr_insts.shape, mr_label.shape))
# Compute the balance of positive/negative count.
p_count = np.sum(mr_label)
logger.info('Default positive percentage in dataset: %f' % (float(p_count) / data_size))
logger.info('Default negative percentage in dataset: %f' % (float(data_size-p_count) / data_size))
# Use 0.7/0.3 partition of the whole data.
num_classes = 2
num_train = int(data_size * 0.7)
num_test = data_size - num_train
logger.info("Training set size = {}, test set size = {}".format(num_train, num_test))
train_insts, train_labels = mr_insts[:num_train, :, :], to_categorical(mr_label[:num_train, :], num_classes)
test_insts, test_labels = mr_insts[num_train:, :, :], to_categorical(mr_label[num_train:, :], num_classes)
# Initialize model training configuration
learn_rate = args.rate
batch_size = args.size
epoch = args.epoch
# Number of hidden units in RNN.
num_hidden = 100
# Different model to use, training phase.
# Your code here: use the following handler model to name your Keras model.
model = None
if args.model == "rnn":
    # You should implement your vanilla RNN + mean pooling here.
    model = Sequential()
    # input_shape = (None, num_feats), where the first None means the length of the sequence and the 
    # second num_feats means the number of features. 
    model.add(SimpleRNN(num_hidden, input_shape=(None, word_embedding.embedding_dim()), return_sequences=True))
    # Mean pooling along time dimension.
    model.add(GlobalAveragePooling1D())
    # Logistic regression classification.
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
elif args.model == "gru":
    # You should implement your GRU + mean pooling here.
    model = Sequential()
    # input_shape = (None, num_feats), where the first None means the length of the sequence and the 
    # second num_feats means the number of features. 
    model.add(GRU(num_hidden, input_shape=(None, word_embedding.embedding_dim()), return_sequences=True))
    # Mean pooling along time dimension.
    model.add(GlobalAveragePooling1D())
    # Logistic regression classification.
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])    
elif args.model == "lstm":
    # You should implement your LSTM + mean pooling here.
    model = Sequential()
    # input_shape = (None, num_feats), where the first None means the length of the sequence and the 
    # second num_feats means the number of features. 
    model.add(LSTM(num_hidden, input_shape=(None, word_embedding.embedding_dim()), return_sequences=True))
    # Mean pooling along time dimension.
    model.add(GlobalAveragePooling1D())
    # Logistic regression classification.
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])    
else:
    raise NameError("{} model not supported.".format(args.model))

# Training phase begins.
logger.info("Finish building model: {}".format(model.summary()))
start_time = time.time()
model.fit(train_insts, train_labels, epochs=epoch, batch_size=batch_size, verbose=2)
end_time = time.time()
logger.info("Time used for training the model = {} seconds.".format(end_time - start_time))
# Test phase.
_, acc = model.evaluate(test_insts, test_labels, batch_size=batch_size, verbose=2)
logger.info("Sentiment analysis accuracy with {} on test set = {}".format(args.model, acc))
