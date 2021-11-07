#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:14:34 2021

@author: leonl42

Unit test for testing the abbrevation replacement in the preprocessing pipeline

"""

from scripts.preprocessing.abbrevations import Abbrevations
from scripts.util import COLUMN_TWEET
import unittest
import pandas as pd

class AbbrevationsTest(unittest.TestCase):
    """Test if abbrevations are replaced correctly"""
    
    def setUp(self):
        
        self._df = pd.DataFrame()
        self._df[COLUMN_TWEET] = ["I won’t be me i’ll be someone who doesn’t didn’t do"]
        self._expected_output = "I will not be me i will be someone who does not did not do"
        
        self._abbrevations = Abbrevations()
        
    def test_abbrevation_replacement(self):
        """Test abbrevation replacement on a predefined string"""
        
        without_abbrevations = self._abbrevations.fit_transform(self._df)
        self.assertEqual(without_abbrevations[COLUMN_TWEET][0], self._expected_output)
        
        

if __name__ == '__main__':
    unittest.main()