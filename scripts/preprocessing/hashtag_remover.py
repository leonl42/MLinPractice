#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 21:02:37 2021

@author: leonl42
"""

from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_TWEET
import re

class HashtagRemover(Preprocessor):
    """Remove every sequence of letters which starts with a '#' from the tweet"""
    
    def __init__(self):
        """Initialize the Hashtag remover with the given input and output column."""
        super().__init__([COLUMN_TWEET], COLUMN_TWEET)
        
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Remove Hashtags from the tweet"""
        
        # initialize column to return
        new_column = []
        
        for tweet in inputs[0]:
            
            # use regular expressions to remove sequence of letters that start with a hashtag.
            # It is assumed that there exists a space after the hashtag, because otherwise
            # words which don't belong to the hashtag would be removed
            tweet_without_hashtags = re.sub(r" ?#[^\s]*", "",tweet)
            new_column.append(tweet_without_hashtags)
        
                
        
        return new_column
    
