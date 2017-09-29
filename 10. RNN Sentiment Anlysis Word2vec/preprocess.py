#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-09-10
# @Author  : Ivan
import os, sys
import re
import logging
import csv
import time
import random
import io
# Set the basic configuration of the logging system
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')
logger = logging.getLogger(__name__)

mr_positive_filename = 'rt-polarity.pos'
mr_negative_filename = 'rt-polarity.neg'
# positive and negative lists
mr_positive_list, mr_negative_list = [], []
# Processing positive file
with open(mr_positive_filename,"r",errors='ignore') as fin:
    for line in fin:
        words = line.split()
        words = [word.lower() for word in words]
        review = ' '.join(words)
        review = review.replace('-', ' ')
        review = ''.join([ch for ch in review if ch.isalpha() or ch == ' '])
        words = review.split()
        words = filter(lambda x: len(x) > 1, words)
        review = ' '.join(words)
        mr_positive_list.append(review)
# Processing negative file
with io.open(mr_negative_filename, 'r',errors='ignore') as fin:
    for line in fin:
        words = line.split()
        words = [word.lower() for word in words]
        review = ' '.join(words)
        review = review.replace('-', ' ')
        review = ''.join([ch for ch in review if ch.isalpha() or ch == ' '])
        words = review.split()
        words = filter(lambda x: len(x) > 1, words)
        review = ' '.join(words)
        mr_negative_list.append(review)
# Random shuffle
pos_index = list(range(len(mr_positive_list)))
neg_index = list(range(len(mr_negative_list)))
random.shuffle(pos_index)
random.shuffle(neg_index)
mr_positive_list = list(map(lambda x: mr_positive_list[x], pos_index))
mr_negative_list = list(map(lambda x: mr_negative_list[x], neg_index))

with open('mr-polarity.pos', 'w') as fout:
    for line in mr_positive_list:
        fout.writelines(line+'\n')
with open('mr-polarity.neg', 'w') as fout:
    for line in mr_negative_list:
        fout.writelines(line+'\n')
