#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:36:58 2021

@author: leonl42

Unit test for testing the ngram feature ectraction

"""

import unittest
import pandas as pd
from scripts.feature_extraction.feature_ngrams import FeatureNGrams
from scripts.util import COLUMN_TWEET,SUFFIX_POST
from numpy.testing import assert_array_equal
class NGramsTest(unittest.TestCase):  
    """"Test the ngram feature"""
    
    def test_bigrams(self):
        """Test if ngram feature correctly counts bigrams and returns the correct feature vector"""
        
        df = pd.DataFrame()
        df[COLUMN_TWEET+SUFFIX_POST] = [[("i", "smth"),("am", "smth"),("viral", "smth")],
                                       [("i", "smth"),("am", "smth"),("not", "smth"),("viral", "smth")],[]]
        expected_output = ([[1,1],[1,0],[0,0]])
        
        bigrams = FeatureNGrams(2,2,COLUMN_TWEET+SUFFIX_POST)
        feature_vector = bigrams.fit_transform(df)
        
        assert_array_equal(feature_vector, expected_output)
        
    def test_threegrams(self):
        """Test if ngram feature correctly counts trigrams and returns the correct feature vector"""
        
        df = pd.DataFrame()
        df[COLUMN_TWEET] = ["This is tweet number one and not two",
                            "This is in fact not a tweet and not two",
                            " "]
        expected_output = ([[1,1,1,1,1,1],[1,0,0,0,0,0],[0,0,0,0,0,0]])
        
        bigrams = FeatureNGrams(3,6,COLUMN_TWEET)
        feature_vector = bigrams.fit_transform(df)
        
        assert_array_equal(feature_vector, expected_output)


if __name__ == '__main__':
    unittest.main()