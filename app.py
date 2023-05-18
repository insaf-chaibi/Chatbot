from flask import Flask, request ,render_template
from chatbot import chatbot_response, symptoms

app = Flask(__name__,template_folder='template')

@app.route('/', methods=['GET'])
def getindex():
      return render_template("index.html")


@app.route('/chatbot', methods=['POST'])
def send_response():
    request_data = request.get_json()
    msg = request_data['msg']
    print(msg)
    return chatbot_response(msg)

@app.route('/symptoms', methods=['GET'])
def send_symp():
     return "symptoms"

if __name__ == '__main__':
    app.run(debug=True, port=8002)

