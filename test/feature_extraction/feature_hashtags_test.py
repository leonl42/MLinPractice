#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 22:28:47 2021

@author: leonl42

Unit test for testing the hashtag feature

"""


import unittest
import pandas as pd
import numpy as np
from scripts.feature_extraction.feature_hashtags import FeatureHashtags
from scripts.util import COLUMN_HASHTAGS
from numpy.testing import assert_array_equal

class HashtagsFeatureTest(unittest.TestCase):
    """Test if one hot encoding for hashtags works fine"""
    
    def setUp(self):
        """"specify necessary variables"""
        
        self._df = pd.DataFrame()
        self._df[COLUMN_HASHTAGS] = [["hi","hallo"],['machinelearning'],[],["test"]]
        self._expected_column = np.array([[1,1,0,0],[0,0,1,0],[0,0,0,0],[0,0,0,1]])
        self._feature_hashtags = FeatureHashtags(4)
        

    def test_conversion(self):
        assert_array_equal(self._feature_hashtags.fit_transform(self._df), self._expected_column)
        

if __name__ == '__main__':
    unittest.main()