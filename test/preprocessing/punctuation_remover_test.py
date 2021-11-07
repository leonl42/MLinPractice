#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:03:06 2021

@author: leonl42

Unit test for testing if punctuation removal works correctly

"""

from scripts.preprocessing.punctuation_remover import PunctuationRemover
from scripts.util import COLUMN_TWEET
import unittest
import pandas as pd

class PunctuationRemoverTest(unittest.TestCase):
    """"Test punctuation removal"""
    
    def setUp(self):
        
        self._df = pd.DataFrame()
        self._df[COLUMN_TWEET] = ["This,, tweet##()() has a lot of !!punctuation%%"]
        self._expected_output = "This tweet has a lot of punctuation"
        self._punctuation_remover = PunctuationRemover()
        
    def test_punctuation_removal(self):
        """"Test punctuation removal on a predefined string"""
        
        without_puncuation = self._punctuation_remover.fit_transform(self._df)
        self.assertEqual(without_puncuation[COLUMN_TWEET][0], self._expected_output)



if __name__ == '__main__':
    unittest.main()

