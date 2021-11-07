#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:22:23 2021

@author: leonl42

Unit test for testing the lemmatization of words

"""

from scripts.preprocessing.lemmatization import Lemmatization
from scripts.util import COLUMN_TWEET,SUFFIX_POST
import unittest
import pandas as pd

class LemmatizationTest(unittest.TestCase):
    """Test lemmatization"""
    
    def setUp(self):
        
        self._df = pd.DataFrame()
        self._df[COLUMN_TWEET+SUFFIX_POST] = [[("I", "PRP"),("went", "VBD"),("viral", "JJ")]]
        self._expected_output = [("I", "PRP"),("go", "VBD"),("viral", "JJ")]
        
        self._lemmatizer = Lemmatization()
        
    def test_lemmatization(self):
        """Test if all words are lemmatized correctly"""
        
        lemmatized_output = self._lemmatizer.fit_transform(self._df)
        self.assertEqual(lemmatized_output[COLUMN_TWEET+SUFFIX_POST][0], self._expected_output)
        

if __name__ == '__main__':
    unittest.main()