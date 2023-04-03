# SIL775-Project
This is a project on biometric matching of ears using neural network

feature_extractor.py - Extracts the edges and angles from the BGR (png) images

recognizer.py - Neural Network for classification of edge images

sad.py generates the interclass and intraclass SAD scores for all possible pairs in the organized dataset.

Matcher - takes two input images and returns whether they belong to the same person

matcher.py - Neural Network using SAD training data

matcher2.py - Neural Network matcher for randomly selected training data

