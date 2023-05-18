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

#symptoms = [0 for i in range(len(df1.columns[1:-1]))]

'''
def chatbot_response():
    isInit = False
    isRunning = True

    while(isRunning):
        if(not isInit):
            ints = predict_class("", model)
            (res, is_diagnosis) = getResponse(ints, intents)
            print(res)
            isInit = True

        else:
            msg = input("")
            ints = predict_class(msg, model)
            (res, is_diagnosis) = getResponse(ints, intents)

            if(is_diagnosis):
                print(symptoms)

                print(type(str(predict(symptoms))))
                isRunning = False
            else:
                print(res)'''
'''
def chatbot_response():
    responses = []
    isInit = False
    isRunning = True

    while isRunning:
        if not isInit:
            ints = predict_class("", model)
            (res, is_diagnosis) = getResponse(ints, intents)
            responses.append(res)
            isInit = True

        else:
            msg = input("")
            ints = predict_class(msg, model)
            (res, is_diagnosis) = getResponse(ints, intents)

            if is_diagnosis:
                responses.append(str(predict(symptoms)))
                isRunning = False
            else:
                responses.append(res)

    return responses
'''

'''
def chatbot_response(msg):
    ints = predict_class(msg, model)
    (res, is_diagnosis) = getResponse(ints, intents)
    if(is_diagnosis):
        ch=str(predict(symptoms))
        return ({"result":ch})
    else:
        return ({"result":res})
'''
    
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


print("Doggy wants to help you figure out if your dog has severe health issues that require an urgent visit to the vet.")
print("Do you want to give it a shot?")
print("Doggy would be happy to help you out")
'''
while True:
    print("")
    msg = input("")
    ints = predict_class(msg,model)
    res = getResponse(ints, intents)
    print(res)
'''

data = pd.read_csv(r"dataset2.csv")
data = data.iloc[:, 1:]

def get_tag(value, intents_json):
    list_of_intents = intents_json["intents"]
    result = None  # Set an initial value for result

    for i in list_of_intents:
        if value in i["patterns"]:
            result = i["tag"]
            break
    return result

def chatbot_decision_tree():
    tree = severity_model.tree_
    feature_names = data.columns[:-1]
    node = 0

    while tree.children_left[node] != -1:
        feature = tree.feature[node]
        feature_name = feature_names[feature]
        threshold = tree.threshold[node]
        value = input(f"Does your dog have {feature_name}? ")

        ints = predict_class(value, model)
        res = get_tag(ints, intents)

        if res == 'Positive':
            node = tree.children_right[node]
        elif res == 'Negative':
            node = tree.children_left[node]
        else:
            print("Invalid input. Please answer with valid inputs.")
            continue


    severity = severity_model.classes_[tree.value[node].argmax()]

    print(f"The predicted disease severity is {severity}.\nPlease consult a vet for further evaluation.")

chatbot_decision_tree()
