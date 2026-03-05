import requests
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

BASE_URL = "http://localhost:8080"

answer_prompt = """Act as a technical document analyst. Your goal is to give a clear and concise answer to the recruiter by providing relevant information and put forward the Maxime profile."""

query = """Act as a technical document analyst. Your goal is to distill the provided text into a high-density, structured summary for a RAG indexing system.

        Instructions:

        Core Context: Start with a single sentence defining the specific scope of this fragment.

        Key Technical Data: Extract and list all architectures ,specific parameters, and methodologies.

        Critical Findings: Summarize results, observations (e.g., entropy trends), or limitations (e.g., redundancy, noise) mentioned.

        Constraints: Precision: Do NOT omit numerical data or specific technical names.

        Concision: Use bullet points. Avoid conversational filler like 'the document discusses' or 'the author says'.

        Self-Containment: Ensure the summary is understandable without needing the full original text"""

query2 = """summarize the following text in one sentence. MANDATORY: Keep as possible technical keywords. These words are keys for RAG retrieval. Use no more than 30 words."""


def envoyer_capsule(nom_fichier):
    """
    
    """
    loader = PyPDFLoader(f"documents/{nom_fichier}")
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])
    try:
        # On vise la route du Gateway
        requests.post(f"{BASE_URL}/post", json={"contenu": full_text, "nom_fichier": nom_fichier[:-4], "query": query, "query2": query2})
        print(f" Réponse : OK")
    except Exception as e:
        print(f" Erreur : {e}")

def chat_request(message):
    time = datetime.now().strftime("%A %d %B %Y, %H:%M")
    prompt = f""""
        DATE ACTUELLE: {time}, Act as a technical document analyst. Your goal is to give a clear and concise answer to the recruiter by providing relevant information and put forward the Maxime profile.
        """

    data = {"query": f"[The recruiter question]: {message}", "prompt": prompt}
    try:
        response = requests.post(f"{BASE_URL}/chat", json=data)
        response.raise_for_status()
        result = response.json().get("answer", "Pas de réponse")
        # promptt_used = response.json().get("context_used", "Pas de contexte")
        print(result)
    except Exception as e:
        print(f" Erreur: {e}")


if __name__ == "__main__":

    #envoyer_capsule("Mémoire_Carbon_calculator.pdf")

    chat_request("""When did Maxime start working at Oh green?""")
