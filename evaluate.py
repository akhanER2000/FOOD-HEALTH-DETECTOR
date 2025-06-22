import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parámetros
MODEL_PATH = "foodnet_model.h5"
DATA_DIR = "data"
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# Preparar datos (sólo validación)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Cargar modelo
model = load_model(MODEL_PATH)

# Predicciones
pred_probs = model.predict(val_generator)
y_pred = (pred_probs > 0.5).astype(int).flatten()
y_true = val_generator.classes

# Reporte de métricas
print("\n📊 Classification Report:\n")
target_names = list(val_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.show()
