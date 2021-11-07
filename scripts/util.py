#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

from datetime import datetime
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import OneHotEncoder
from multipledispatch import dispatch
import pandas as pd
from nltk import FreqDist

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"
COLUMN_TIMEZONE = "timezone"
COLUMN_DATE = "date"
COLUMN_TIME = "time"
COLUMN_TIMEZONE = "timezone"
COLUMN_HASHTAGS = "hashtags"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_TIMEDELTAS = "timedeltas"

SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_POST = "_post"

# dtype arguments for pandas
PANDAS_DTYPE = {COLUMN_TIMEZONE : str}

# constant for filling empty arrays when hot encoding because
# an array which has arrays of different sizes as elements wont be interpreted as a 2-dimensional array but
# rather as a 1 dimensional array which has lists as elements
EMPTY = "Empty/#f4903mM=Upü3ümr3##9uMP)ur3mr3w9po8(Z79m3owr#+fuwpm9f+3p9f#3wmufw3ü9f#####fafuman8zr8z83o"

# twitter creation date. This is the reference date when calculating the timedeltas
TWITTER_CREATION_YEAR = datetime.strptime("2006", '%Y')
TWITTER_CREATION_DATE = datetime.strptime("03-21", '%m-%d')
TWITTER_CREATION_TIME = datetime.strptime("00:00:00", '%H:%M:%S')

# implement a simple one hot encoder
def one_hot_encoding(categories, to_encode, condition):
    """"default (self-made) one hot encoding."""
    
    matrix = []
    for count1,element in enumerate(to_encode):
        encoding_vector = []
        for count2,category in enumerate(categories):
            
            # check if category fulfills condition given the element to encode
            # if the condition is fullfilled a 1 will be appended, if not, a 0
            encoding_vector.append(int(condition(category,element)))
        matrix.append(encoding_vector)
    return np.array(matrix)

def prepare_data_for_one_hot_encoder(input_column):
    """Prepare data for the sklearn one hot encoder"""
    
    # initialize column to return
    column = []
    
    for count,elements in enumerate(input_column):
        as_python_variable = literal_eval(str(elements))
        
        # check the data type of the row in the data column.
        # In the end we want an array with the shape (-1,1).
        # So if it is a list, we append each element as an array to our column.
        # If it is the empty list we return the EMPTY constant a an array.
        # If it is a single variable, we return this variable as a list.
        if type(as_python_variable) == list:
            if len(as_python_variable) == 0:
                column.append([EMPTY])
            else:
                for element in as_python_variable:
                    column.append([element])
        else:
            column.append([as_python_variable])
        
    return column

def get_freq_dist(column):
    """"calculate frequency distribution given a column from the data frame"""
    
    freq_dist = FreqDist()
    
    for count, row in enumerate(column):
        
        as_python_variable = literal_eval(str(row))
        
        # check if rows are lists or simple strings
        # Depending on what the rows are, either iterate through them
        # Or just process the raw string
        if type(as_python_variable) == list:
            for element in as_python_variable:
                freq_dist[element] += 1
        else:
            freq_dist[as_python_variable] += 1
    
    return freq_dist
            
def join_feature_rows(row1,row2,index):
    """Join feature vectors with the logical or"""
    
    return np.logical_or(row1, row2[index])
    


def sklearn_one_hot_encoding_fit(column_prepared, one_hot_encoder):
    """"call fit on a given sklearn one_hot_enccoder"""
    
    one_hot_encoder.fit(column_prepared)

@dispatch(pd.Series, OneHotEncoder)
def sklearn_one_hot_encoding_transform(column, one_hot_encoder):
    """one hot encode with not prepared data"""
    
    column_prepared = prepare_data_for_one_hot_encoder(column)
    return sklearn_one_hot_encoding_transform(column, column_prepared, one_hot_encoder)

@dispatch(pd.Series, list, OneHotEncoder)
def sklearn_one_hot_encoding_transform(column, column_prepared ,one_hot_encoder):
    """"one hot encode with prepared data"""
    
    encoded_array = one_hot_encoder.transform(column_prepared)
    num_categories = len(one_hot_encoder.categories_)
    cur_encoder_row = 0
    reshaped_encoded_array = np.array([])
    
    # one_hot_encoder.transform returns an 2 dimensional array, where each element from each list from the dataframe
    # has its own feature vector.
    # But we want feature vectors for the list of elements in our dataframe.
    # To get this, we logically join the feature vectors from each element in a list
    for row in column:
        new_row = np.zeros(num_categories)
        
        as_python_variable = literal_eval(str(row))
        
        if type(as_python_variable) == list:
            if len(as_python_variable) == 0:
                
                # logically join feature vectors and select feature vector of next element from our encoded list
                new_row = join_feature_rows(new_row, encoded_array, cur_encoder_row)
                cur_encoder_row += 1
    
            else:
                for y in range(len(literal_eval(str(row)))):
                    
                    # logically join feature vectors and select feature vector of next element from our encoded list
                    new_row = join_feature_rows(new_row, encoded_array, cur_encoder_row)
                    cur_encoder_row += 1
                    
        else:
            
            # logically join feature vectors and select feature vector of next element from our encoded list
            new_row = join_feature_rows(new_row, encoded_array, cur_encoder_row)
            cur_encoder_row += 1
            
        reshaped_encoded_array = np.append(reshaped_encoded_array,new_row)
    
    # reshape encoded array such that each row from our original data column
    # has its own feature vector containing all features
    reshaped_encoded_array = np.reshape(reshaped_encoded_array,(column.shape[0],-1))
    
    return reshaped_encoded_array
        
            
    


    