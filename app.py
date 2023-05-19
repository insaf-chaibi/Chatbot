from flask import Flask, request ,render_template
from chatbot import chatbot_decision_tree, opening_message

app = Flask(__name__,template_folder='template')

@app.route('/', methods=['GET'])
def home():
      return render_template("index.html")

@app.route('/chatbot', methods=['POST'])
def send_response():
    request_data = request.get_json()
    msg = request_data['msg']

    return chatbot_response(msg)

@app.route('/', methods=['GET'])
def bot_response():
    return opening_message()

if __name__ == '__main__':
    app.run(debug=True, port=8002)
