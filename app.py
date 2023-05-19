from flask import Flask, request ,render_template, jsonify
from chatbot import chatbot_decision_tree, opening_message
import pandas as pd
import json
import joblib
app = Flask(__name__,template_folder='template')

@app.route('/', methods=['GET'])
def home():
      return render_template("index.html")

@app.route('/send', methods=["GET", "POST"])
def send_response():
    request_data = request.get_json()
    msg = request_data['msg']
    return chatbot_decision_tree(msg)

@app.route('/chat', methods=["GET", "POST"])
def chat_response():
    return opening_message()


data = pd.read_csv("dataset2.csv")
data = data.iloc[:, 1:]

intents = json.loads(open("intents2.json").read())

Severity_model = joblib.load("model/severity_model.joblib")


opening_msg = "Hi, I'm a bot. I can help you diagnose your dog's condition. What is your dog's name?"
conversation_state = {
    "request_id" : {
        "severity_node" : 0,
        "asked_symptoms" : []
    }
}

@app.route('/get_all_the_symptoms', methods=["GET", "POST"])
def get_all_the_symptoms():
    return jsonify(list(data.columns))

@app.route('/diagnose', methods=["GET", "POST"])
def diagnose():
    """
        1- get the request_id and answer from the request
        2- check if the request_id is in the conversation_state
        3- add the request_id to the conversation_state with severity_node = 0
        4- start asking the questions one by one and add the answers to the conversation_state
        5- if the answer is yes, go to the right child
        6- if the answer is no, go to the left child
            till the severity_node == -1
        7- if the answer is not yes or no, return an error
    """
    request_id , answer = request.json.get("request_id" , None) , request.json.get("answer" , None)
    if not request_id :
        return jsonify({"response" : "request_id not found"}) , 400
    if not answer :
        return jsonify({"response" : "answer not found in the request"}) , 400
    
    # answer == reset
    if answer.lower() == "reset":
        conversation_state.pop(request_id , None)
        return jsonify({"response" : "the conversation state is reset"}) , 200
    
    if answer.lower() not in ["yes" , "no"]:
        # reset the conversation state
        conversation_state.pop(request_id , None)
        return jsonify({"response" : "answer must be 'yes' or 'no'"}) , 400
    
    if request_id not in conversation_state :
        conversation_state[request_id] = {"severity_node" : 0 , "asked_symptoms" : []}
    
    severity_node = conversation_state[request_id]["severity_node"]
    severity_tree = Severity_model.tree_
    feature_names = data.columns[:-1]

    if severity_tree.children_left[severity_node] == -1:
        return jsonify({"response" : "the diagnosis is already made type 'reset' to start over"}) , 200
    
    feature = severity_tree.feature[severity_node]
    feature_name = feature_names[feature]
    threshold = severity_tree.threshold[severity_node]
    
    # if it's the first asked symptoms return the opening message
    asked_symptoms = conversation_state[request_id]["asked_symptoms"]
    if len(asked_symptoms) == 0:
        conversation_state[request_id]["asked_symptoms"].append(feature_name)
        return jsonify({"response" : f"Does your dog have {feature_name}?"})
    
    if answer.lower() == 'yes':
        severity_node = severity_tree.children_right[severity_node]
    elif answer.lower() == 'no':
        severity_node = severity_tree.children_left[severity_node]
    
    conversation_state[request_id]["severity_node"] = severity_node
    
    if severity_tree.children_left[severity_node] == -1:
        severity = Severity_model.classes_[severity_tree.value[severity_node].argmax()]
        return jsonify({"response" : f"The diagnosis is {severity}.\nPlease consult a vet for further evaluation."}) , 200
    
    feature = severity_tree.feature[severity_node]
    feature_name = feature_names[feature]
    
    return jsonify({"response" : f"Does your dog have {feature_name}?"})

@app.route('/reset', methods=["GET", "POST"])
def reset():
    """
        1- get the request_id from the request
        2- check if the request_id is in the conversation_state
        3- remove the request_id from the conversation_state
    """
    request_id = request.json.get("request_id" , None)
    if not request_id :
        return jsonify({"response" : "request_id not found"}) , 400
    if request_id not in conversation_state :
        return jsonify({"response" : "request_id not found"}) , 400
    conversation_state.pop(request_id)
    return jsonify({"response" : "the conversation state is reset successfully"})
    
    
if __name__ == '__main__':
    app.run(debug=True, port=8002)
