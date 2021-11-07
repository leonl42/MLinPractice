#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on un Oct Nov 23:24:00 2021

@author: leonl42

Unit test for testing the timedelta format, not if the conversion was correct. 

"""

from scripts.preprocessing.timedeltas import Timedeltas
from scripts.util import COLUMN_DATE, COLUMN_TIME, COLUMN_TIMEZONE, COLUMN_TIMEDELTAS
import unittest
import pandas as pd

class TimedeltasTest(unittest.TestCase):
    """Doesn't test if the values are calcuated correctly, but rather if the output format is correct"""
    
    def setUp(self):
        
        self._df = pd.DataFrame()

        self._df[COLUMN_DATE] = ["2021-04-14"]
        self._df[COLUMN_TIME] = ["04:46:42"]
        self._df[COLUMN_TIMEZONE] = ["+0530"]
        
        self._timedeltas = Timedeltas()
        
    def test_timedelta_format(self):
        """Test if the new column has the correct format"""
        
        timedeltas = self._timedeltas.fit_transform(self._df)
        
        self.assertEqual(type(timedeltas[COLUMN_TIMEDELTAS][0][0]), float)
        self.assertEqual(type(timedeltas[COLUMN_TIMEDELTAS][0][1]), float)
        self.assertEqual(type(timedeltas[COLUMN_TIMEDELTAS][0][2]), float)
        
        
if __name__ == '__main__':
    unittest.main()