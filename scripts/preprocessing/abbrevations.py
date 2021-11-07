#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 23:38:21 2021

@author: leonl42

Preprocessor for replacing english abbrevations with their full (long) form.

"""

from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_TWEET

class Abbrevations(Preprocessor):
    """Replace abbrevations with their corresponding long form"""
    
    def __init__(self):
        # will just overwrite original tweet column
        super().__init__([COLUMN_TWEET], COLUMN_TWEET)
        
    def _set_variables(self, inputs):
        """"set abbrevations"""
        self._abbrevations = [("isn’t","is not"),("won’t","will not"),("doesn’t","does not"),("hasn’t","has not"),("haven’t","have not"),("didn’t","did not"),
                              ("there’s", "there is"),("i’ll", "i will")]
        
    def _get_values(self, inputs):
        """"replace abbrevations"""

        for abbrevation in self._abbrevations:
            # iterate through the tweets and replace the abbrevations
            inputs[0] = [tweet.replace(abbrevation[0],abbrevation[1]) for tweet in inputs[0]]
        
        return inputs[0]
        
        