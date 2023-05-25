# STRIDE Classifier

This is an implementation of a STRIDE classifier, which is used to classify security threats in software systems in one of the STRIDE categories. 

It uses scikit-multilearn to perform multi-label classification.

STRIDE is a widely used framework for categorizing threats and stands for Spoofing, Tampering, Repudiation, Information disclosure, Denial of service, and Elevation of privilege.

## Installation

To install the required dependencies for this project, run the following command:

```
pip install -r requirements.txt
```

This will install all the necessary packages listed in the `requirements.txt` file.

## Data 

The CAPEC attack pattern collection in its version 3.9 has 559 active elements. Since the number of elements is so small the best classifiers to use seem to be Naive Bayes and SVC. In this project I tried to test how different the accuracy is when using different techniques plus the different estimators from scikit-multilearn for multi-label classification: Binary Relevance, Classifier Chain and Label Powerset.

## Usage

There are four scripts available:
* capec_to_excel.py extracts and combines both the Ostering's CAPEC-STRIDE mapping and the information from CAPEC (Current version is 3.9).
* stride_classifier_model_builder.py creates the classifier model
* general_stride_classifier.py is a demo script to play with. It waits for a user input.
* mitre_attack_stride_classifier.py uses the model to infer which STRIDE category would fit best every Mitre ATT&CK Framework technique

None of them have arguments.

Additionally, there are some examples about how to use scikit-learn and other interesting machine learning techniques inside the "examples" folder.


## Improvements

* Establish the final objective of this project
* Find better ways to preprocess the input
* Add more features from the original CAPEC file
* Find the best hyperparameters for each type of model to increase accuracy and F1-score