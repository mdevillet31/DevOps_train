#!/bin/bash

cd "$(dirname "$0")/.."

echo "--- Nettoyage des anciens conteneurs ---"
docker compose down

echo "--- Reconstruction et Lancement ---"
docker compose up --build -d

echo "--- Statut des services ---"
docker compose ps

echo "--- Logs en temps réel (Ctrl+C pour quitter l'affichage, le serveur restera allumé) ---"
docker compose logs -f