#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:58:31 2021

@author: leonl42

Uses one hot encoding to specify which hashtags were used in a tweet

"""
import numpy as np
from scripts.feature_extraction.feature_extractor import FeatureExtractor
from scripts.util import COLUMN_HASHTAGS, sklearn_one_hot_encoding_transform,get_freq_dist
from sklearn.preprocessing import OneHotEncoder



class FeatureHashtags(FeatureExtractor):
    """Create a feature vector for each tweet that comprises which of the most common hashtags were used in this tweet"""
    
    # constructor
    def __init__(self, num_hashtags):
        self._num_hashtags = num_hashtags
        super().__init__([COLUMN_HASHTAGS], ["{0}_hashtags_{1}".format(COLUMN_HASHTAGS,i) for i in range(num_hashtags)])
        
    def _set_variables(self, inputs):
        """"determine all hashtags which were used in the dataset"""
        
        # determine frequency distribution of all hashtags and take the most common hashtags
        freq_dist = get_freq_dist(inputs[0])
        most_common = np.array([element[0] for element in freq_dist.most_common(self._num_hashtags)])
        
        most_common = most_common.reshape(-1,1)
        self._one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse = False)
        self._one_hot_encoder.fit(most_common)
        
        # rename each feature dimension according to its corresponding hashtag
        self._feature_name = ["{0}_{1}".format(COLUMN_HASHTAGS, hashtag) for hashtag in most_common]
        
    def _get_values(self, inputs):
        """"do one hot encoding for hashtags"""

        return sklearn_one_hot_encoding_transform(inputs[0], self._one_hot_encoder)
        
        

