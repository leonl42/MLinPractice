#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 12:42:46 2021

@author: leonl42

Preprocessor for applying lemmatization to all words in the tweet.
Lemmatization is the task of finding a common lemma for a word.
Exptects tweet to be part of speech tagged. Will overwrite part of speech column
"""

from scripts.preprocessing.preprocessor import Preprocessor
from scripts.util import COLUMN_TWEET, SUFFIX_POST
from nltk.stem import 	WordNetLemmatizer
from ast import literal_eval

class Lemmatization(Preprocessor):
    """Lemmatized word depending on their wordnet position"""
    
    def __init__(self):
        
        #will just overwrite old part of speech tagged column
        super().__init__([COLUMN_TWEET+SUFFIX_POST], COLUMN_TWEET+SUFFIX_POST)
        
    #no need to set internal variables
    
    def _get_wordnet_pos(self, treebank_tag):
        """get the according wordnet position from the word tag"""
        
        if treebank_tag.startswith('J'):
            return 'a'
        elif treebank_tag.startswith('V'):
            return 'v'
        elif treebank_tag.startswith('N'):
            return 'n'
        elif treebank_tag.startswith('R'):
            return 'r'
        else:
            return ''
    
        
    def _get_values(self, inputs):
       """"lemmatize all words"""
       
       wordnet_lemmatizer = WordNetLemmatizer()
       column = []

       for list_of_words in inputs[0]:
           lemmatized_list_of_words = []
           for word in literal_eval(str(list_of_words)):
                
               # get the according wordnet position from its tag
               wordnet_pos = self._get_wordnet_pos(word[1])
               
               # check if wordnet position is valid and lemmatize acording to that
               lemma = wordnet_lemmatizer.lemmatize(word[0]) if wordnet_pos == '' else wordnet_lemmatizer.lemmatize(word[0],pos=self._get_wordnet_pos(word[1])) 
               lemmatized_list_of_words.append((lemma,word[1]))
                
           column.append(lemmatized_list_of_words)
        
       return column
   

