from PIL import Image
import os

def limpiar_imagenes_corruptas(carpeta):
    total = 0
    borradas = 0
    for root, _, files in os.walk(carpeta):
        for file in files:
            if file.lower().endswith(".jpg"):
                total += 1
                try:
                    img_path = os.path.join(root, file)
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"❌ Imagen corrupta eliminada: {img_path}")
                    os.remove(img_path)
                    borradas += 1
    print(f"\n✅ Completado. {borradas} imágenes corruptas eliminadas de {total}.")

# Ejecutar
if __name__ == "__main__":
    limpiar_imagenes_corruptas("data")
