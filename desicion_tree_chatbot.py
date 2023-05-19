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

Severity_model = joblib.load("model/severity_model.joblib")

def chatbot_decision_tree():
    severity_tree = Severity_model.tree_
    feature_names = data.columns[:-1]
    severity_node = 0
    
    while severity_tree.children_left[severity_node] != -1:
        feature = severity_tree.feature[severity_node]
        feature_name = feature_names[feature]
        threshold = severity_tree.threshold[severity_node]
        value = input(f" ")
        
        if value.lower() == 'yes':
            severity_node = severity_tree.children_right[severity_node]
        elif value.lower() == 'no':
            severity_node = severity_tree.children_left[severity_node]
        else:
            print("Invalid input. Please answer with 'yes' or 'no'.")
            continue
            
    severity = Severity_model.classes_[severity_tree.value[severity_node].argmax()]

    print(f"The diagnosis is {severity}.\nPlease consult a vet for further evaluation.")


if __name__ == "__main__":
    chatbot_decision_tree()