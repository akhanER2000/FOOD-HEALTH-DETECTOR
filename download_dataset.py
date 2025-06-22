import os
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# --- Configuraciones ---
# Este es el nombre del dataset en Kaggle: usuario/nombre-dataset
DATASET_ID = "kmader/food41"
# Carpeta donde se guardará el dataset descomprimido
DEST_DIR = Path("food-101")

# --- Lógica de Descarga y Descompresión ---
if DEST_DIR.exists():
    print(f"✅ El directorio '{DEST_DIR}' ya existe. Se omite la descarga.")
else:
    print(f"Iniciando la descarga del dataset '{DATASET_ID}' desde Kaggle...")
    
    try:
        # Inicializar la API de Kaggle
        api = KaggleApi()
        api.authenticate()

        # Crear el directorio de destino
        DEST_DIR.mkdir(parents=True, exist_ok=True)

        # Descargar los archivos del dataset
        api.dataset_download_files(DATASET_ID, path=DEST_DIR, unzip=True)
        
        print(f"✅ Dataset descargado y descomprimido exitosamente en '{DEST_DIR}'.")

    except Exception as e:
        print(f"❌ Error durante la descarga: {e}")
        print("Asegúrate de que tus credenciales de Kaggle ('kaggle.json') están configuradas correctamente.")
        print("Y que has aceptado las reglas del dataset en la página de Kaggle, si es necesario.")