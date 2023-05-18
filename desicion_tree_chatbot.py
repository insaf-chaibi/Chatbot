import pandas as pd
import numpy as np
import joblib
import sklearn.tree._tree 
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import json

data = pd.read_csv(r"dataset2.csv")
data = data.iloc[:, 1:]

intents = json.loads(open("intents2.json").read())
severity_model = joblib.load("model/severity_model.joblib")

def get_tag(value, intents):
    for intent in intents:
        patterns = intent["patterns"]
        if value.lower() in patterns:
            return intent["tag"]
    return 'Invalid'

def chatbot_decision_tree():
    tree = severity_model.tree_
    feature_names = data.columns[:-1]
    node = 0

    while tree.children_left[node] != -1:
        feature = tree.feature[node]
        feature_name = feature_names[feature]
        threshold = tree.threshold[node]
        value = input(f"Does your dog have {feature_name}? ")

        tag = get_tag(value, intents)
        if tag == 'Positive':
            node = tree.children_right[node]
        elif tag == 'Negative':
            node = tree.children_left[node]
        else:
            print("Invalid input. Please answer with valid inputs.")
            continue


    severity = severity_model.classes_[tree.value[node].argmax()]

    print(f"The predicted disease severity is {severity}.\nPlease consult a vet for further evaluation.")

chatbot_decision_tree()
