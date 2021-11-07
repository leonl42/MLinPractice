#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 12:59:53 2021

@author: leonl42

Preprocessor for part of speech tagging the tweet.
The tweet has to be tokenized.

"""

from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_TWEET, SUFFIX_TOKENIZED, SUFFIX_POST
from nltk import pos_tag
from ast import literal_eval

class Post(Preprocessor):
    """Part of speech tags the tokenized tweet"""
    
    def __init__(self):
        
        #create a new column for the tagged tweet
        super().__init__([COLUMN_TWEET+SUFFIX_TOKENIZED], COLUMN_TWEET+SUFFIX_POST)
        
    #no need to set internal variables
    
    def _get_values(self, inputs):
        """tag the words"""
        
        column = []
        
        for list_of_words in inputs[0]:
            
            #tag each list of tokenized words
            column.append(pos_tag(literal_eval(str(list_of_words))))
              
        return column
    
