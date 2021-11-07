#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:08:40 2021

@author: leonl42

Unit test for testing the lowercase conversion in the preprocessing pipeline

"""

import unittest
import pandas as pd
from scripts.preprocessing.lower import Lower
from scripts.util import COLUMN_TWEET

class LowerTest(unittest.TestCase):
    """"Test the lowercase preprocessing step"""
    
    def setUp(self):
        self._df = pd.DataFrame()
    
        # make one random string and copy it, but replace every upper letter with the corresponding lower one in this copy
        _string_to_test = "I WHISH thath this ##2#E220md STRING becomes LoWerCase ÄÜÖÄ"
        self._expected_result = "i whish thath this ##2#e220md string becomes lowercase äüöä"
        
        self._df[COLUMN_TWEET] = [_string_to_test]
        self._lower = Lower()
        
    def test_lowercase(self):
        """Test lowercase conversion on a predefined string"""
        
        lowercase_string = self._lower.fit_transform(self._df)
        self.assertEqual(lowercase_string[COLUMN_TWEET][0], self._expected_result)
        

if __name__ == "__main__":
    unittest.main()