import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from foodnet_model import build_foodnet # Importamos desde el archivo local

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Parámetros generales
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 50 # Aumentamos las épocas, EarlyStopping decidirá cuándo parar
LEARNING_RATE = 1e-4
DATA_DIR = "data"
MODEL_PATH = "foodnet_model.h5"

# --- Preparación de Datos ---
# El aumento de datos ya se hizo offline. Aquí solo reescalamos y dividimos.
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # 20% de los datos para validación
)

# Generadores de datos
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# --- Cálculo de Pesos de Clase (Class Weights) ---
# Esto es crucial para que el modelo preste más atención a la clase con menos ejemplos
# incluso después del aumento, ya que la división puede no ser perfecta.
counter = np.bincount(train_generator.classes)
num_no_saludable = counter[train_generator.class_indices['no_saludable']]
num_saludable = counter[train_generator.class_indices['saludable']]
total_samples = num_saludable + num_no_saludable

print(f"Muestras de entrenamiento: No Saludable = {num_no_saludable}, Saludable = {num_saludable}")

if num_saludable > 0 and num_no_saludable > 0:
    weight_for_no_saludable = (1 / num_no_saludable) * (total_samples / 2.0)
    weight_for_saludable = (1 / num_saludable) * (total_samples / 2.0)
    class_weights = {
        train_generator.class_indices['no_saludable']: weight_for_no_saludable,
        train_generator.class_indices['saludable']: weight_for_saludable
    }
    print(f"Pesos de clase calculados: {class_weights}")
else:
    class_weights = None
    print("No se pudieron calcular los pesos de clase.")

# --- Construcción y Compilación del Modelo ---
model = build_foodnet(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True)
]

# --- Entrenamiento ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights # ¡Aplicamos los pesos de clase aquí!
)

print(f"\n✅ Entrenamiento finalizado. Modelo guardado como {MODEL_PATH}")

# --- Visualización de Resultados ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.savefig('training_history.png')
plt.show()