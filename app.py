from flask import Flask, request ,render_template
from chatbot import chatbot_decision_tree, opening_message

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


if __name__ == '__main__':
    app.run(debug=True, port=8002)
