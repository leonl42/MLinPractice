#!/bin/bash
# overall pipeline for the ML experiments

echo "loading data"
scripts/load_data.sh
echo "preprocessing"
scripts/preprocessing.sh
echo "feature extraction"
scripts/feature_extraction.sh
echo "dimensionality reduction"
scripts/dimensionality_reduction.sh
echo "classification"
scripts/classification.sh