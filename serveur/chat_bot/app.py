import os
import psycopg2
from flask import Flask, jsonify, request
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
import litellm


app = Flask(__name__)
QDRANT_URL = "http://qdrant-server:6333"
COLLECTION_NAME = "ma_capsule_perso"
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://ollama-server:11434")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama-server:11434",
)

@app.route("/chat", methods=["POST"])
def ask_microservice():
    """
    Endpoint to receive user queries and forward them to the chatbot service. Expects a JSON payload with the following structure:
    {
        "query": "The user's query or prompt to be processed by the chatbot.",
        "prompt": "Additional instructions or context for the chatbot to consider when generating a response."
    }
    The endpoint will forward the query and prompt to the chatbot service, which will process the input and return a response. The gateway will then return the chatbot's response to the user.
    """
    answere = request.get_json()
    query = answere.get("query", "")
    prompt = answere.get("prompt", "")

    if not query:
        return jsonify({"error": "Aucune question fournie"}), 400
    
    try:
        client = QdrantClient(url=QDRANT_URL)
        vector = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
        docs = vector.similarity_search(query, k=5)

        context = "\n".join([f"Nom du projet: {d.metadata.get('source', 'Inconnue')}\n CONTENU:{d.metadata.get('full text', 'Inconnu')}" for d in docs])

        system_prompt = f"""{prompt}CONTEXTE:\n{context}"""

        response = litellm.completion(
            model="ollama/mistral-nemo:12b",
            api_base="http://ollama-server:11434",
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
        )
        return jsonify({
            "answer": response.choices[0].message.content,
            "context_used": context
        })
    
    except Exception as e:
        return f"Erreur de communication : {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
