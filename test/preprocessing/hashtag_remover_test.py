#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 22:23:51 2021

@author: leonl42
"""


from scripts.preprocessing.hashtag_remover import HashtagRemover
from scripts.util import COLUMN_TWEET
import unittest
import pandas as pd

class HashtagRemoverTest(unittest.TestCase):
    """Test if hashtags are removed correctly from the tweet"""
    
    def setUp(self):
        
        self._df = pd.DataFrame()
        self._df[COLUMN_TWEET] = ["This #me is a test tweet #everything is cool ####manyhashtags"]
        self._expected_output = "This is a test tweet is cool"
        
        self.HashtagRemover = HashtagRemover()
        
    def test_hashtag_removal(self):
        
        hashtags_removed = self.HashtagRemover.fit_transform(self._df)
        self.assertEqual(hashtags_removed[COLUMN_TWEET][0], self._expected_output)
        
        

if __name__ == '__main__':
    unittest.main()