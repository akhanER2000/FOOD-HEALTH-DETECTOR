import os
import shutil
from pathlib import Path

# Diccionario con la clasificación (sin cambios)
class_metadata = {
    "apple_pie": 1, "baby_back_ribs": 1, "baklava": 1, "beef_carpaccio": 0,
    "beef_tartare": 0, "beet_salad": 0, "beignets": 1, "bibimbap": 0,
    "bread_pudding": 1, "breakfast_burrito": 1, "bruschetta": 0, "caesar_salad": 1,
    "cannoli": 1, "caprese_salad": 0, "carrot_cake": 1, "ceviche": 0,
    "cheesecake": 1, "cheese_plate": 1, "chicken_curry": 0, "chicken_quesadilla": 1,
    "chicken_wings": 1, "chocolate_cake": 1, "chocolate_mousse": 1, "churros": 1,
    "clam_chowder": 1, "club_sandwich": 1, "crab_cakes": 1, "creme_brulee": 1,
    "croque_madame": 1, "cup_cakes": 1, "deviled_eggs": 1, "donuts": 1,
    "dumplings": 1, "edamame": 0, "eggs_benedict": 1, "escargots": 0,
    "falafel": 0, "filet_mignon": 0, "fish_and_chips": 1, "foie_gras": 1,
    "french_fries": 1, "french_onion_soup": 1, "french_toast": 1,
    "fried_calamari": 1, "fried_rice": 1, "frozen_yogurt": 1, "garlic_bread": 1,
    "gnocchi": 1, "greek_salad": 0, "grilled_cheese_sandwich": 1, "grilled_salmon": 0,
    "guacamole": 0, "gyoza": 1, "hamburger": 1, "hot_and_sour_soup": 0,
    "hot_dog": 1, "huevos_rancheros": 0, "hummus": 0, "ice_cream": 1,
    "lasagna": 1, "lobster_bisque": 1, "lobster_roll_sandwich": 1,
    "macaroni_and_cheese": 1, "macarons": 1, "miso_soup": 1, "mussels": 0,
    "nachos": 1, "omelette": 0, "onion_rings": 1, "oysters": 0,
    "pad_thai": 1, "paella": 0, "pancakes": 1, "panna_cotta": 1,
    "peking_duck": 1, "pho": 0, "pizza": 1, "pork_chop": 0, "poutine": 1,
    "prime_rib": 1, "pulled_pork_sandwich": 1, "ramen": 1, "ravioli": 1,
    "red_velvet_cake": 1, "risotto": 1, "samosa": 1, "sashimi": 0,
    "scallops": 0, "seaweed_salad": 0, "shrimp_and_grits": 1,
    "spaghetti_bolognese": 1, "spaghetti_carbonara": 1, "spring_rolls": 0,
    "steak": 0, "strawberry_shortcake": 1, "sushi": 0, "tacos": 0,
    "takoyaki": 1, "tiramisu": 1, "tuna_tartare": 0, "waffles": 1
}

# --- Configuraciones Ajustadas ---
# La API de Kaggle descomprime en una estructura más simple.
# La carpeta de imágenes ahora está en 'food-101/images'
SRC_DIR = Path("food-101/images") 
DEST_DIR = Path("data")
MAX_IMAGES_PER_CLASS = 1000  # Usamos todas las imágenes disponibles

# Crear carpetas destino
(DEST_DIR / "saludable").mkdir(parents=True, exist_ok=True)
(DEST_DIR / "no_saludable").mkdir(parents=True, exist_ok=True)

# Procesar clases
for class_name, label in class_metadata.items():
    src_class_dir = SRC_DIR / class_name
    if not src_class_dir.exists():
        print(f"⚠️  Clase no encontrada: {class_name}")
        continue

    dest_label_dir = DEST_DIR / ("saludable" if label == 0 else "no_saludable")
    
    # Se cambió la lógica para copiar todos los archivos sin renombrar para simplicidad
    count = 0
    for img_file in src_class_dir.glob("*.jpg"):
        # Copiamos el archivo manteniendo su nombre original para evitar colisiones
        shutil.copy(img_file, dest_label_dir / img_file.name)
        count += 1

    print(f"✅ {class_name}: {count} imágenes copiadas a {'saludable' if label == 0 else 'no_saludable'}")

print("\n🎉 Dataset reorganizado exitosamente.")