#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:09:20 2021

@author: leonl42

Computes the ngram distribution and the corresponding feature vectors by looking
at the most frequent ngrams and using one hot encoding.

"""


from scripts.feature_extraction.feature_extractor import FeatureExtractor
from scripts.util import COLUMN_TWEET,SUFFIX_TOKENIZED,SUFFIX_POST, one_hot_encoding,get_freq_dist
from nltk import ngrams
from ast import literal_eval



class FeatureNGrams(FeatureExtractor):
    """Computes the ngram distribution and the corresponding feature vectors"""
    
    def __init__(self, n, num_ngrams, input_column):
        
        self._n = n
        self._num_ngrams = num_ngrams
        self._input_column = input_column
        super().__init__([input_column], ["{0}_ngrams_{1}".format(input_column,i) for i in range(num_ngrams)])
        
    def _prepare_data_for_ngram_model(self, inputs):
        """"Fit the data into a predefined format"""
        
        formatted_column = []
        
        # because the different preprocessing steps formate the original tweet different, 
        # we have to distinguish between 3 inputs:
        # the original tweet
        if self._input_columns == [COLUMN_TWEET]:
            for tweet in inputs[0]:
                formatted_column.append(str(tweet).split())
        
        #the tweet tokenized
        elif self._input_columns == [COLUMN_TWEET+SUFFIX_TOKENIZED]:
            
            for tokenized_tweet in inputs[0]:
                words = [str(word) for word in literal_eval(str(tokenized_tweet))]
                formatted_column.append(words)
        
        # the tweet pos tagged
        elif self._input_columns == [COLUMN_TWEET+SUFFIX_POST]:
            
            for post_tweet in inputs[0]:
                words = [str(word_and_tag[0]) for word_and_tag in literal_eval(str(post_tweet))]
                formatted_column.append(words)
        
        return formatted_column
    
    def _set_variables(self, inputs):
        """"Determine most common ngrams in the tweets"""
        
        formatted_column = self._prepare_data_for_ngram_model(inputs)
        self._ngrams = []
        
        for row in formatted_column:
            ngrams_zipped = (ngrams(row, self._n))
            for unziped in ngrams_zipped:
                
                #convert the n-tuple representing a ngram into a string and append this
                #string as a list to our ngrams
                self._ngrams.append([' '.join(list(unziped))])
            
        # compute the frequency distribution of our ngrams
        # and select the most common
        freq_dist = get_freq_dist(self._ngrams)
        self._ngrams = [element[0] for element in freq_dist.most_common(self._num_ngrams)]
        
        # rename each feature dimension according to its corresponding ngram
        self._feature_name = ["{0}_{1}".format(self._input_column, ngram) for ngram in self._ngrams]
            
    def _get_values(self, inputs):
        """Determine which ngrams are used in a tweet and compute the corresponding feature vectors"""
        
        formatted_column = self._prepare_data_for_ngram_model(inputs)
        return one_hot_encoding(self._ngrams, formatted_column, lambda c,e: c in ' '.join(e))
