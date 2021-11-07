#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:14:59 2021

@author: leonl42

Unit test for testing if part of speech tagging works correctly
"""

from scripts.preprocessing.post import Post
from scripts.util import COLUMN_TWEET, SUFFIX_TOKENIZED,SUFFIX_POST
import unittest
import pandas as pd

class PostTests(unittest.TestCase):
    """"Test part of speech tagging"""
    
    def setUp(self):
        
        self._df = pd.DataFrame()
        self._df[COLUMN_TWEET+SUFFIX_TOKENIZED] =[["I", "went", "viral"]]
        
        #formulate expected output with the corresponding role of the word
        #part of speech tagger should come up with the same tags
        self._expected_output = [("I", "PRP"),("went", "VBD"),("viral", "JJ")]
        
        self._post = Post()
        
    def test_part_of_speech_tagging(self):
        """Test if part of speech tagger gives the correct output format and tags
        all corresponding words correctly"""
        
        tagged_output = self._post.fit_transform(self._df)
        self.assertEqual(tagged_output[COLUMN_TWEET+SUFFIX_POST][0], self._expected_output)
        
        
if __name__ == '__main__':
    unittest.main()