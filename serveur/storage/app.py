import psycopg2
from flask import Flask, jsonify, request
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from litellm import completion


app = Flask(__name__)
QDRANT_URL = "http://qdrant-server:6333"
COLLECTION_NAME = "ma_capsule_perso"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://ollama-server:11434",
)

def process_chunk(chunk, query):
    """
    use the LLM to process the chunk of text and rewite it in a more concise way, keeping only the most important information.
    """
    try:
        response = completion(
            model="ollama/qwen2.5:7b",
            api_base="http://ollama-server:11434",
            temperature=0.1,
            max_tokens=300,
            messages=[
                {"role": "system", "content": query},
                {"role": "user", "content": chunk.page_content},
            ],
        )
    except Exception as e:
        raise Exception(f"Error during LLM processing: {e}")
    return response.choices[0].message.content

@app.route("/save", methods=["POST"])
def save():
    """
    Entry point for saving content to the vector store. Expects a JSON payload with the following structure:
    {
        "content": "The text content to be processed and saved.",
        "nom_fichier": "optional_name_for_source",
        "query": "Instructions for processing the chunk of text.",
        "query2": "Instructions for summarizing the processed chunk."
    }
    The endpoint will split the content into chunks, process each chunk with the provided query, summarize it with the second query, and then save the results to the Qdrant vector store.
    """
    data = request.json
    if not data or "content" not in data:
        return jsonify({"error": "No content provided"}), 400
    
    content = data.get("content")
    
    try:
        name_project =data.get("nom_fichier", "unknown_project")
        query = data.get("query", "")
        query2 = data.get("query2", "")
    
        docs = [Document(page_content=content)]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_documents(docs)
        
        results = []
        for chunk in enumerate(chunks):
                result = process_chunk(chunk, query)
                summary = process_chunk(Document(page_content=result), query2)
                doc = Document(page_content=summary, metadata={
                    "source": name_project,
                    "chunk_id": i,
                    "full text": result})
                results.append(doc)

        QdrantVectorStore.from_documents(
            results,
            embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )

        return jsonify({"status": "saved", "chunks": len(results)}), 201
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
