# -*- coding: utf-8 -*-

"""
Preprocessor that removes stopwords from the tweet.
Can only be applied if the tweet has been tokenized.
Stopwords are determined by nltk's stopwords for the english language

@author: leonl42
"""


from scripts.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
from scripts.util import COLUMN_TWEET, SUFFIX_TOKENIZED
from ast import literal_eval

class StopWords(Preprocessor):
    """Remove stopwords from the tweet"""
    
    def __init__(self):
        
        # will just overwrite old tokenized column
        super().__init__([COLUMN_TWEET+SUFFIX_TOKENIZED], COLUMN_TWEET+SUFFIX_TOKENIZED)
        
    def _set_variables(self, inputs):
        """get nltks stopwords for the english language and save them"""
        
        self._stopwords = stopwords.words('english')
        
    def _get_values(self, inputs):
        """"remove stopwords from tokenized tweet"""
        
        column= []
        for list_of_words in inputs[0]:
            # create a new list where only words from the old list which are not stopwords appear
            list_without_stopwords = [word for word in literal_eval(str(list_of_words)) if (not word in self._stopwords)]
            column.append(list_without_stopwords)
        
        return column
        
        

        
        