#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple feature that counts the number of characters in the given column.

Created on Wed Sep 29 12:29:25 2021

@author: lbechberger, leonl42
"""

import numpy as np
from scripts.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the character-based length as a feature
class FeatureCharacterLength(FeatureExtractor):
    """Count number of characters in the given column"""
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], ["{0}_charlength".format(input_column)])
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the word length based on the inputs
    def _get_values(self, inputs):
        
        result = []
        for tweet in inputs[0]:
            
            # take care of the case that the value in the column is NaN
            if tweet is not np.nan:
                result.append(len(tweet))
            else:
                result.append(0)
            
        result = np.array(result).reshape(-1,1)
        return result

