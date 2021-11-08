#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train or evaluate a single classifier with its given set of hyperparameters.

Created on Wed Sep 29 14:23:48 2021

@author: lbechberger
"""

import argparse, pickle
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from mlflow import log_metric, log_param, set_tracking_uri

# setting up CLI
parser = argparse.ArgumentParser(description = "Classifier")
parser.add_argument("input_file", help = "path to the input pickle file")
parser.add_argument("-s", '--seed', type = int, help = "seed for the random number generator", default = None)
parser.add_argument("-e", "--export_file", help = "export the trained classifier to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import a trained classifier from the given location", default = None)
parser.add_argument("-m", "--majority", action = "store_true", help = "majority class classifier")
parser.add_argument("-f", "--frequency", action = "store_true", help = "label frequency classifier")
parser.add_argument("--knn", type = int, help = "k nearest neighbor classifier with the specified value of k", default = None)
parser.add_argument("-svm", "--svm", action = "store_true", help = "support vector machine classifier", default = None)
parser.add_argument("--mlp", nargs="+",type = int, help = "multi Layer Perceptron classifier", default = None)
parser.add_argument("--rndmforest", nargs="+", help = "random forest classifier", default = None)
parser.add_argument("-a", "--accuracy", action = "store_true", help = "evaluate using accuracy")
parser.add_argument("-k", "--kappa", action = "store_true", help = "evaluate using Cohen's kappa")
parser.add_argument("-f1", "--f1", action = "store_true", help = "evaluate using F1 score")
parser.add_argument("--log_folder", help = "where to log the mlflow results", default = "data/classification/mlflow")
args = parser.parse_args()

# load data
with open(args.input_file, 'rb') as f_in:
    data = pickle.load(f_in)

set_tracking_uri(args.log_folder)

if args.import_file is not None:
    # import a pre-trained classifier
    with open(args.import_file, 'rb') as f_in:
        input_dict = pickle.load(f_in)
    
    classifier = input_dict["classifier"]
    for param, value in input_dict["params"].items():
        log_param(param, value)
    
    log_param("dataset", "validation")

else:   # manually set up a classifier
    
    if args.majority:
        # majority vote classifier
        print("    majority vote classifier")
        log_param("classifier", "majority")
        params = {"classifier": "majority"}
        classifier = DummyClassifier(strategy = "most_frequent", random_state = args.seed)
        
    elif args.frequency:
        # label frequency classifier
        print("    label frequency classifier")
        log_param("classifier", "frequency")
        params = {"classifier": "frequency"}
        classifier = DummyClassifier(strategy = "stratified", random_state = args.seed)  
    
    elif args.knn is not None:
        print("    {0} nearest neighbor classifier".format(args.knn))
        log_param("classifier", "knn")
        log_param("k", args.knn)
        params = {"classifier": "knn", "k": args.knn}
        standardizer = StandardScaler()
        knn_classifier = KNeighborsClassifier(args.knn, n_jobs = -1)
        classifier = make_pipeline(standardizer, knn_classifier)
    
    elif args.svm:
        print("    support vector machine classifier")
        log_param("classifier", "svm")
        params = {"classifier": "svm"}
        standardizer = StandardScaler()
        classifier = make_pipeline(standardizer,svm.SVC())
    
    elif args.mlp is not None:
      print("    Multi layer perceptron classifier")
      log_param("classifier", "mlp")
      params = {"classifier": "mlp"}
      standardizer = StandardScaler()
      classifier = make_pipeline(standardizer,MLPClassifier(max_iter=args.mlp[0],hidden_layer_sizes = tuple(args.mlp[1:len(args.mlp)])))
    
    elif args.rndmforest is not None:        
       print("    Random Forest classifier")
       log_param("classifier", "rndmforest")
       params = {"classifier": "rndmforest"}
       standardizer = StandardScaler()
       classifier = make_pipeline(standardizer,RandomForestClassifier(n_estimators=int(args.rndmforest[0]),max_depth=int(args.rndmforest[1]),max_features=int(args.rndmforest[2])))
    
    classifier.fit(data["features"], data["labels"].ravel())
    log_param("dataset", "training")

# now classify the given data
prediction = classifier.predict(data["features"])

# collect all evaluation metrics
evaluation_metrics = []
if args.accuracy:
    evaluation_metrics.append(("accuracy", accuracy_score))
if args.kappa:
    evaluation_metrics.append(("Cohen_kappa", cohen_kappa_score))
if args.f1:
    evaluation_metrics.append(("f1_score", f1_score))

# compute and print them
for metric_name, metric in evaluation_metrics:
    metric_value = metric(data["labels"], prediction)
    print("    {0}: {1}".format(metric_name, metric_value))
    log_metric(metric_name, metric_value)
    
# export the trained classifier if the user wants us to do so
if args.export_file is not None:
    output_dict = {"classifier": classifier, "params": params}
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(output_dict, f_out)