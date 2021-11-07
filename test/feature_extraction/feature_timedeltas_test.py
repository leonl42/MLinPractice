#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:43:32 2021

@author: leonl42

Unit test for testing if the timedelta conversion from a pd column to a numpy array works fine

"""

import unittest
import pandas as pd
import numpy as np
from scripts.feature_extraction.feature_timedeltas import FeatureTimedeltas
from scripts.util import COLUMN_TIMEDELTAS
from numpy.testing import assert_array_equal

class TimedeltasFeatureTest(unittest.TestCase):
    """test if the timedelta features are extracted correctly"""
    
    def setUp(self):
        """"specify necessary variables"""
        
        self._df = pd.DataFrame()
        self._df[COLUMN_TIMEDELTAS] = [[4343243,432432432,542543534],[34243254,23425235,64745754]] 
        self._expected_column = np.array([[4343243,432432432,542543534],[34243254,23425235,64745754]])
        self._feature_timedeltas = FeatureTimedeltas(COLUMN_TIMEDELTAS)
        

    def test_conversion(self):
        assert_array_equal(self._feature_timedeltas.transform(self._df), self._expected_column)
        

if __name__ == '__main__':
    unittest.main()