from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

SAVE_URL = "http://storage:5001"
CHAT_URL = "http://chatbot:5002"


@app.route("/")
def home():
    return "Bienvenue sur le Gateway !"


@app.route("/post", methods=["POST"])
def post_message():
    user_data = request.get_json()
    response = requests.post(
        f"{SAVE_URL}/save", json={"content": user_data["contenu"]}, timeout=60
    )
    return jsonify({"gateway_status": "Transmis", "storage_response": response.text})


@app.route("/exec", methods=["POST"])
def execute_SQL():
    user_data = request.get_json()
    response = requests.post(
        f"{SAVE_URL}/exec", json={"content": user_data["query"]}, timeout=60
    )
    return jsonify({"gateway_status": "Transmis", "storage_response": response.text})


@app.route("/messages", methods=["GET"])
def get_messages():
    response = requests.get("http://storage:5001/messages", timeout=60)
    return response.text


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
