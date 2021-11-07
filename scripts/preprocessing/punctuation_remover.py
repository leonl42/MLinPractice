#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the tweet text.
Created on Wed Sep 29 09:45:56 2021
@author: lbechberger, leonl42

"""

import string
from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_TWEET

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class PunctuationRemover(Preprocessor):
    """Preprocessor class for removing punctuation from the tweet"""
    
    # constructor
    def __init__(self):
        # input column "tweet", same output column
        super().__init__([COLUMN_TWEET], COLUMN_TWEET)
    
    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store punctuation for later reference
        self._punctuation = "[{}]".format(string.punctuation + "—")
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        
        new_column = []
        for row in inputs[0]:
            
            #for each punctuation mark, replace it with the empty string in the current tweet
            for punctuation in self._punctuation:
                row = str(row).replace(punctuation, "")
                
            new_column.append(row)
        
        return new_column
    
