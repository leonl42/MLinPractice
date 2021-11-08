#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 22:12:24 2021

@author: leonl42

print useful statistics about the data

"""

import numpy as np
import matplotlib as plt
from scripts.util import get_freq_dist, PANDAS_DTYPE,COLUMN_LABEL,COLUMN_HASHTAGS, COLUMN_TIMEDELTAS,SUFFIX_POST,COLUMN_TWEET, COLUMN_LIKES,COLUMN_RETWEETS
import pandas as pd
import csv
from nltk import ngrams
from ast import literal_eval

# load data
df = pd.read_csv("data/preprocessing/preprocessed.csv", quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n", dtype = PANDAS_DTYPE)

################################################
#####plot a pair of frequency distributions#####
################################################

def plot_freq_dist(freq_dist_pos,freq_dist_neg,num, c1, c2):
    
    fig = plt.pyplot.figure()
    ax = fig.add_axes([0,0,1,1])
    X = np.arange(num)
    
    # ectract relative frequency of the 'num' most common elements and store them in the data vector
    data = [[freq_dist_pos.freq(element[0])for element in freq_dist_pos.most_common(num)],
            [freq_dist_neg.freq(element[0]) for element in freq_dist_pos.most_common(num)]]
    
    # extract the most common elements and store their labels
    labels = [element[0] for element in freq_dist_pos.most_common(num)]
    
    ax.set_xticklabels(labels, rotation=90)
    ax.xaxis.set_ticks([i for i in range(num)])
    ax.bar(X + 0.00, data[0], color = c1, width = 0.25)
    ax.bar(X + 0.25, data[1], color = c2, width = 0.25)
    
    plt.pyplot.show()
    

###########################################################
#####Split dataframe into positive and negative labels#####
###########################################################

grouped = df.groupby(COLUMN_LABEL)
df_true = grouped.get_group(True)
df_false= grouped.get_group(False)


#################################################
#####Plot most common hashtags per dataframe#####
#################################################

def get_most_common_hashtags(df):
    freq_dist = get_freq_dist(df[COLUMN_HASHTAGS])
    return freq_dist

freq_dist_hashtags_pos = get_most_common_hashtags(df_true)
freq_dist_hashtags_neg = get_most_common_hashtags(df_false)

plot_freq_dist(freq_dist_hashtags_pos,freq_dist_hashtags_neg,50,'g','r')
plot_freq_dist(freq_dist_hashtags_neg,freq_dist_hashtags_pos,50,'r','g')

#################################
#####Plot average time delta#####
#################################

def statistics_time_deltas(df):
    year = np.array([])
    date = np.array([])
    time = np.array([])
    
    for entry in (df[COLUMN_TIMEDELTAS]):
        entry = literal_eval(str(entry))
        year = np.append(year,entry[0])
        date = np.append(date,entry[1])
        time = np.append(time,entry[2])
    
    return [(np.mean(year),np.std(year)),(np.mean(date),np.std(date)),(np.mean(time),np.std(time))]

print(statistics_time_deltas(df_true))
print(statistics_time_deltas(df_false))


############################
#####Plot average ngram#####
############################

def average_ngram(df,n):
    formatted_column = []
    for post_tweet in df[COLUMN_TWEET+SUFFIX_POST]:
        words = [str(word_and_tag[0]) for word_and_tag in literal_eval(str(post_tweet))]
        formatted_column.append(words)
    ngrams_list = []
    
    for row in formatted_column:

        ngrams_zipped = (ngrams(row, n))
        for unziped in ngrams_zipped:
            
            #convert the n-tuple representing a ngram into a string and append this
            #string as a list to our ngrams
            ngrams_list.append([' '.join(list(unziped))])
        
    #compute the frequency distribution of our ngrams
    #and select the most common
    freq_dist = get_freq_dist(ngrams_list)
    return freq_dist
    #ngrams_list = [element for element in freq_dist.most_common(num_ngrams)]
    #return ngrams_list



freq_dist_ngram_pos = average_ngram(df_true,2)
freq_dist_ngram_neg = average_ngram(df_false,2)

print(freq_dist_ngram_pos.most_common(1))
print(freq_dist_ngram_neg.most_common(1))

plot_freq_dist(freq_dist_ngram_pos,freq_dist_ngram_neg,20,'g','r')
plot_freq_dist(freq_dist_ngram_neg,freq_dist_ngram_pos,20,'r','g')

################################
#####Plot label distribution####
################################

def get_label_dist_is_viral(df, threshold):
    df[COLUMN_LABEL] = (df[COLUMN_LIKES] + df[COLUMN_RETWEETS]) > threshold
    grouped = df.groupby(COLUMN_LABEL)
    df_true = grouped.get_group(True)
    return len(df_true)

num_pos_list = []
thresholds = []
iterations = 500

for i in range(iterations):
    num_pos_list.append(get_label_dist_is_viral(df, i))
    thresholds.append(i)

plt.pyplot.scatter(thresholds,num_pos_list)

###########################
#####character lenght######
###########################

def mean_std_tweet_length(tweets):
    arr = []
    for tweet in tweets:
        arr.append(len(str(tweet)))
        
    arr = np.array(arr)
    return (np.mean(arr),np.std(arr))

print(mean_std_tweet_length(df_true[COLUMN_TWEET]))
print(mean_std_tweet_length(df_false[COLUMN_TWEET]))

######################################################################
#####plot how many viral tweets use a certain number of hashtags######
######################################################################

num_hashtags = {}
for hashtag_list in df_true[COLUMN_HASHTAGS]:
    for hashtag in literal_eval(str(hashtag_list)):
        
        #lenght of the hashtag is the number of hashtags that were used
        num = len(hashtag)
        if num in num_hashtags:
            num_hashtags[num] +=1
        else:
            num_hashtags[num] =1
            
plt.pyplot.scatter(num_hashtags.keys(),num_hashtags.values())    
            
        

