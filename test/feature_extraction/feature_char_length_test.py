#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 8 01:02:00 2021

@author: leonl42

Unit test for the character length feature

"""

import unittest
import pandas as pd
from scripts.feature_extraction.feature_character_length import FeatureCharacterLength
from scripts.util import COLUMN_TWEET
from numpy.testing import assert_array_equal

class CharLengthTest(unittest.TestCase):  
    """Test if character length is calculated correctly"""
    
    def test_char_length(self):
        """First case: Input is a normal sentence"""
        
        df = pd.DataFrame()
        df[COLUMN_TWEET] = ["char length"]
        expected_output = 11
        
        char_length = FeatureCharacterLength(COLUMN_TWEET)
        feature_vector = char_length.fit_transform(df)
        
        assert_array_equal(feature_vector, expected_output)
    
    def test_case_nan(self):
        """Second case: Input is NaN"""
        
        df = pd.DataFrame()
        df[COLUMN_TWEET] = []
        expected_output = 0
        
        char_length = FeatureCharacterLength(COLUMN_TWEET)
        feature_vector = char_length.fit_transform(df)
        
        assert_array_equal(feature_vector, expected_output)


if __name__ == '__main__':
    unittest.main()