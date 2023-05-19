import random
import numpy as np
import pandas as pd
import pickle
import json
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import joblib
import sklearn.tree._tree 
from sklearn import tree

from severity_train import df1, predict

lemmatizer = WordNetLemmatizer()

model = load_model("model/chatbot_model.h5")
severity_model = joblib.load("model/severity_model.joblib")

intents = json.loads(open("intents2.json").read())

words = pickle.load(open("model/words.pkl", "rb"))
classes = pickle.load(open("model/classes.pkl", "rb"))

# chat functionalities
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    bow = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

data = pd.read_csv(r"dataset2.csv")
data = data.iloc[:, 1:] 

def opening_message(user_input):
        # The chatbot initiates the conversation with the opening message
        opening_message = "Hi, I'm Doggy.\nDoggy wants to help you figure out if your dog has severe health issues that require an urgent visit to the vet. \nDo you want to give it a shot?"
        response = {"result": opening_message}
        return response
    elif user_input is not None:
        ints = predict_class(user_input, model)
        if ints[0]["intent"] == 'Positive':
            response = chatbot_decision_tree(user_input)
            return response
        elif ints[0]["intent"] == 'Negative':
            response = {"result": "Doggy would always be happy to help you out. Welcome anytime :)"}
            return response
        else:
            response = {"result": "Invalid input. Please answer with valid inputs."}
            return response

def chatbot_decision_tree(user_input):
    tree = severity_model.tree_
    feature_names = data.columns[:-1]
    node = 0

    while tree.children_left[node] != -1:
        feature = tree.feature[node]
        feature_name = feature_names[feature]
        threshold = tree.threshold[node]
        #value = input(f"Does your dog have {feature_name}? ")

        # Include the question in the response
        response = {"question": f"Does your dog have {feature_name}?"}

        ints = predict_class(user_input, model)
        
        if ints[0]["intent"] == 'Positive':
            node = tree.children_right[node]
        elif ints[0]["intent"] == 'Negative':
            node = tree.children_left[node]
        else:
            return "Invalid input. Please answer with valid inputs."
            continue

    severity = severity_model.classes_[tree.value[node].argmax()]
    response["result"] = f"The predicted disease severity is {severity}.\nPlease consult a vet for further evaluation."

    return response

#chatbot_decision_tree(user_input)

