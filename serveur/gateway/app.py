from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

SAVE_URL = "http://storage:5001"
CHAT_URL = "http://chatbot:5002"


@app.route("/")
def home():
    return "Welcome to the Gateway !"

@app.route("/post", methods=["POST"])
def post_message():
    """
    Endpoint to receive user data and forward it to the storage service. Expects a JSON payload with the following structure:
    {
        "contenu": "The text content to be processed and saved.",
        "nom_fichier": "optional_name_for_source",
        "query": "Instructions for processing the chunk of text.",
        "query2": "Instructions for summarizing the processed chunk."
    }
    The endpoint will forward the content and associated metadata to the storage service, which will handle the processing and saving of the data.
    """
    user_data = request.get_json()
    response = requests.post(
        f"{SAVE_URL}/save", json={"content": user_data["contenu"], "nom_fichier": user_data["nom_fichier"], "query": user_data["query"], "query2": user_data["query2"]}, timeout=1000
    )
    return jsonify({"gateway_status": "Transmis", "storage_response": response.text})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint to receive user queries and forward them to the chatbot service. Expects a JSON payload with the following structure:
    {
        "query": "The user's query or prompt to be processed by the chatbot.",
        "prompt": "Additional instructions or context for the chatbot to consider when generating a response."
    }
    The endpoint will forward the query and prompt to the chatbot service, which will process the input and return a response. The gateway will then return the chatbot's response to the user.
    """
    user_data = request.get_json()
    response = requests.post(
        f"{CHAT_URL}/chat", json={"query": user_data["query"], "prompt": user_data["prompt"]}, timeout=60
    )
    return response.text


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="5000")
