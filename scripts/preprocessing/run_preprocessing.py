#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from scripts.preprocessing.punctuation_remover import PunctuationRemover
from scripts.preprocessing.tokenizer import Tokenizer
from scripts.preprocessing.hashtag_remover import HashtagRemover
from scripts.preprocessing.lower import Lower
from scripts.preprocessing.abbrevations import Abbrevations
from scripts.preprocessing.timedeltas import Timedeltas
from scripts.util import COLUMN_TWEET, SUFFIX_TOKENIZED, PANDAS_DTYPE

# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TWEET)
parser.add_argument("-tdeltas", "--timedeltas", action = "store_true", help = "create timedeltas for tweet creation datetime")
parser.add_argument("-l", "--lower", action = "store_true", help = "make every letter in the tweet lowercase")
parser.add_argument("-ab", "--abbrevations", action = "store_true", help = "replace abbrevations with their long form")
parser.add_argument("-hr","--hashtag_removal", action = "store_true", help = "remove hashtags from the tweet")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n", dtype = PANDAS_DTYPE)

# collect all preprocessors
preprocessors = []
if args.hashtag_removal:
    preprocessors.append(HashtagRemover())
if args.punctuation:
    preprocessors.append(PunctuationRemover())
if args.lower:
    preprocessors.append(Lower())
if args.abbrevations:
    preprocessors.append(Abbrevations())
if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
if args.timedeltas:
    preprocessors.append(Timedeltas())

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)