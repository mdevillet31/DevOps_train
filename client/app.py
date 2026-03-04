import requests
import os
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

BASE_URL = "http://localhost:8080"
CHAT_URL = "http://localhost:5002/chat"

def envoyer_capsule(texte):
    print(f"Envoi : {texte}")
    try:
        # On vise la route du Gateway
        requests.post(f"{BASE_URL}/post", json={"contenu": texte})
        print(f" Réponse : OK")
    except Exception as e:
        print(f" Erreur : {e}")


def lire_tout():
    print("\n Consultation de la base de données...")
    response = requests.get(f"{BASE_URL}/messages")
    print(f"Données reçues : {response.text}")


def chat_request(message):
    time = datetime.now().strftime("%A %d %B %Y, %H:%M")
    prompt = f""""
        DATE ACTUELLE: {time}"""

    data = {"query": message, "prompt": prompt}
    try:
        response = requests.post(CHAT_URL, json=data)
        response.raise_for_status()
        result = response.json()["answer"]
        print(result)
    except Exception as e:
        print(f" Erreur: {e}")


if __name__ == "__main__":
    #loader = PyPDFLoader("")
    # pages = loader.load()
    # full_text = "\n".join([p.page_content for p in pages])
    # envoyer_capsule(full_text)
    
    # lire_tout()
    chat_request("gives me some exemples of words after lematization")
