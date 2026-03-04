import psycopg2
from flask import Flask, jsonify, request
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings

app = Flask(__name__)
QDRANT_URL = "http://qdrant-server:6333"
COLLECTION_NAME = "ma_capsule_perso"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama-server:11434",
)


def get_db_connection():
    return psycopg2.connect(
        host="db", database="capsule_db", user="user", password="password123"
    )


@app.route("/save", methods=["POST"])
def save():
    content = request.json.get("content")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (content) VALUES (%s)", (content,))
    conn.commit()

    docs = [Document(page_content=content)]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)
    

    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    cur.close()
    conn.close()
    return jsonify({"status": "Sauvegardé"}), 201


@app.route("/messages", methods=["GET"])
def get_messages():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT content FROM messages;")
    row = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return row, 200


@app.route("/exec", methods=["POST"])
def execute_SQL():
    context = request.json.get("content")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(context)
    row = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({"status": f"action efféctué {context} resultat: {row}"}), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
