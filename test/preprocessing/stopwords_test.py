#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:35:57 2021

@author: leonl42

Unit test for testing if the stopword removal works correctly

"""

import unittest
import pandas as pd
from scripts.preprocessing.stopwords import StopWords
from scripts.util import COLUMN_TWEET, SUFFIX_TOKENIZED

class StopWordsTest(unittest.TestCase):
    """"Test the stopword removal"""
    
    def setUp(self):
        self._df = pd.DataFrame() 
        
        # set up a tokenized string for stopword removal
        # and specifiy the expected output (which strings should be removed)
        self._df[COLUMN_TWEET+SUFFIX_TOKENIZED] = [["the", "stopwords","should","be","removed","thiS","shouLdn't"]]
        self._expected_output = ["stopwords","removed", "thiS","shouLdn't"]
        
        self._stopwords = StopWords()
    
    def test_stopword_removal(self):
        """Just does simple stopword removal and tests for the expected output"""
        
        without_stopwords = self._stopwords.fit_transform(self._df)
        self.assertEqual(without_stopwords[COLUMN_TWEET+SUFFIX_TOKENIZED][0], self._expected_output)
        

if __name__ == '__main__':
    unittest.main()        


