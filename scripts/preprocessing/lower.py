#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:31:23 2021

@author: leonl42

Convert every letter in the tweet to lowercase

"""


from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_TWEET

class Lower(Preprocessor):
    """Preprocessor for converting tweet to lowercase"""
    
    def __init__(self):
        # will just overwrite old tweet column
        super().__init__([COLUMN_TWEET], COLUMN_TWEET)
        
    # no need to set internal variables
        
    
    def _get_values(self, inputs):
        """call lower() on all strings"""
        
        return [entry.lower() for entry in inputs[0]]
        


