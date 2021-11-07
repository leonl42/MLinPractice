#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:34:55 2021

@author: leonl42

Feature extracting for converting the already calculated timedeltas into a feature
"""

import numpy as np
from scripts.feature_extraction.feature_extractor import FeatureExtractor
from ast import literal_eval


class FeatureTimedeltas(FeatureExtractor):
    """"Converts the already preprocessed timedeltas into a feature"""
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], ["{0}_timedeltas_{1}".format(input_column,i) for i in range(3)])
        
    #no need to set internal variables
    
    def _get_values(self, inputs):
        """"convert timedeltas into numpy array"""
        
        column = []
        
        for vector in inputs[0]:
            column.append(literal_eval(str(vector)))
            
        return np.array(column)
        