#!/bin/bash

# create directory if not yet existing
mkdir -p data/preprocessing/split/

# install all NLTK models
python -m nltk.downloader all

# add labels
echo "  creating labels"
python -m scripts.preprocessing.create_labels data/raw/ data/preprocessing/labeled.csv

# other preprocessing (removing punctuation etc.)
echo "  general preprocessing"
python -m scripts.preprocessing.run_preprocessing data/preprocessing/labeled.csv data/preprocessing/preprocessed.csv --punctuation --tokenize -hr -tdeltas -l -ab -sw -post -lemma -e data/preprocessing/pipeline.pickle

# split the data set
echo "  splitting the data set"
python -m scripts.preprocessing.split_data data/preprocessing/preprocessed.csv data/preprocessing/split/ -s 42