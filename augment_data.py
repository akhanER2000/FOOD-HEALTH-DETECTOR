import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import ImageFile

# Permite cargar imágenes truncadas o corruptas sin lanzar un error fatal
ImageFile.LOAD_TRUNCATED_IMAGES = True

print("--- Iniciando el Proceso de Aumento de Datos para la Clase Minoritaria ---")

# Rutas
DATA_DIR = "data"
SALUDABLE_DIR = os.path.join(DATA_DIR, "saludable")
NO_SALUDABLE_DIR = os.path.join(DATA_DIR, "no_saludable")

# Contar imágenes iniciales
num_saludable = len(os.listdir(SALUDABLE_DIR))
num_no_saludable = len(os.listdir(NO_SALUDABLE_DIR))

print(f"Imágenes iniciales en 'saludable': {num_saludable}")
print(f"Imágenes iniciales en 'no_saludable': {num_no_saludable}")

# Determinar la clase minoritaria y la mayoritaria
if num_saludable < num_no_saludable:
    minority_dir = SALUDABLE_DIR
    target_count = num_no_saludable
    minority_class_name = 'saludable'
else:
    # En este caso, no se necesita aumento o el desbalance es inverso.
    # Para este proyecto, sabemos que 'saludable' es la minoritaria.
    print("No se requiere aumento o el desbalance es inverso. Saliendo.")
    exit()

num_images_to_generate = target_count - len(os.listdir(minority_dir))
if num_images_to_generate <= 0:
    print("Las clases ya están balanceadas. No se generarán nuevas imágenes.")
    exit()

print(f"Objetivo: Balancear la clase '{minority_class_name}' para que tenga aprox. {target_count} imágenes.")
print(f"Se generarán {num_images_to_generate} imágenes nuevas.")

# Configuración del generador para aumentar imágenes
augment_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Listar las imágenes originales de la clase minoritaria
original_images = [os.path.join(minority_dir, f) for f in os.listdir(minority_dir) if f.endswith('.jpg')]
total_originals = len(original_images)

# Calcular cuántas imágenes aumentadas generar por cada imagen original
if total_originals > 0:
    augmentations_per_image = round(num_images_to_generate / total_originals)
    print(f"Generando aproximadamente {augmentations_per_image} imágenes aumentadas por cada imagen original.")
else:
    print("No hay imágenes en la carpeta de la clase minoritaria para aumentar.")
    exit()

# Generar y guardar las imágenes aumentadas
generated_count = 0
for image_path in original_images:
    try:
        img = load_img(image_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in augment_datagen.flow(x, batch_size=1,
                                          save_to_dir=minority_dir,
                                          save_prefix='aug_',
                                          save_format='jpg'):
            i += 1
            generated_count += 1
            if i >= augmentations_per_image:
                break # Salir para pasar a la siguiente imagen original
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")

print(f"\nProceso completado. Se generaron {generated_count} imágenes nuevas.")
print(f"Total final en 'saludable': {len(os.listdir(SALUDABLE_DIR))}")
print(f"Total final en 'no_saludable': {len(os.listdir(NO_SALUDABLE_DIR))}")