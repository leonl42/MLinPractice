#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from scripts.feature_extraction.feature_character_length import FeatureCharacterLength
from scripts.feature_extraction.feature_collector import FeatureCollector
from scripts.feature_extraction.feature_timedeltas import FeatureTimedeltas
from scripts.feature_extraction.feature_hashtags import FeatureHashtags
from scripts.feature_extraction.feature_ngrams import FeatureNGrams
from scripts.util import COLUMN_TWEET, COLUMN_LABEL, PANDAS_DTYPE, COLUMN_TIMEDELTAS, SUFFIX_POST


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("-t", "--timedeltas", action = "store_true", help = "convert preprocessed timedeltas into a numpy array")
parser.add_argument("--hashtags", type = int, help = "number of most frequent hashtags to use for one hot encoding")
parser.add_argument("--ngrams", nargs = '+', help = "use most common ngrams for one hot encoding")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n", dtype = PANDAS_DTYPE)

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(FeatureCharacterLength(COLUMN_TWEET))
    if args.timedeltas:
        features.append(FeatureTimedeltas(COLUMN_TIMEDELTAS))
    if args.hashtags is not None:
        features.append(FeatureHashtags(args.hashtags))
    if args.ngrams is not None:
        features.append(FeatureNGrams(int(args.ngrams[0]),int(args.ngrams[1]),COLUMN_TWEET+SUFFIX_POST))
    
    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)
    
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)

# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)