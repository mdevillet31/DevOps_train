import psycopg2
from flask import Flask, jsonify, request

# avant from langchain_community.vectorstores import Chroma
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaEmbeddings
import litellm


app = Flask(__name__)
QDRANT_URL = "http://qdrant-server:6333"
COLLECTION_NAME = "ma_capsule_perso"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama-server:11434",
)


@app.route("/chat", methods=["POST"])
def ask_microservice():
    answere = request.get_json()
    query = answere.get("query", "")
    prompt = answere.get("prompt", "")
    if not query:
        return jsonify({"error": "Aucune question fournie"}), 400
    try:

        client = QdrantClient(url=QDRANT_URL)
        vector = QdrantVectorStore(
            client=client, collection_name=COLLECTION_NAME, embedding=embeddings
        )

        docs = vector.similarity_search(query, k=3)

        context = "\n---\n".join([d.page_content for d in docs])

        system_prompt = f"""{prompt}
        CONTEXTE:
        {context}
        """

        response = litellm.completion(
            model="gemini/gemini-2.5-flash",
            # api_base="http://ollama-server:11434",
            # temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )
        return jsonify({
            "answer": response.choices[0].message.content,
            "context_used": context  # On renvoie la liste des extraits trouvés
        })
    except Exception as e:
        return f"Erreur de communication : {e}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
