# pyright: reportMissingImports=false
from tensorflow.keras import layers, models

def build_foodnet(input_shape=(64, 64, 3)):
    """
    Construye el modelo CNN 'FoodNet'.
    La arquitectura incluye capas de Normalización por Lotes (BatchNormalization) para
    estabilizar el entrenamiento y capas de Dropout para prevenir el sobreajuste.
    """
    # El bloque del modelo debe estar indentado para pertenecer a la función
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Bloque Convolucional 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque Convolucional 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque Convolucional 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloque de Clasificación
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') # Sigmoid para clasificación binaria
    ])
    
    # La línea de retorno también debe estar indentada
    return model