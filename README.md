# Proyecto de Visión Artificial: Clasificación de Alimentos y Análisis de Comida Saludable

Este proyecto implementa una Red Neuronal Convolucional (CNN) para clasificar imágenes de alimentos en 101 categorías distintas utilizando el dataset Food-41. Adicionalmente, el sistema está diseñado para realizar una clasificación secundaria que determina si el plato de comida identificado es "saludable" o "no saludable".

El proyecto se desarrolla íntegramente en un notebook de Google Colab, abarcando desde la descarga y preprocesamiento de los datos hasta la definición, entrenamiento y evaluación del modelo.

## 📜 Tabla de Contenidos

1.  [Descripción del Proyecto](https://www.google.com/search?q=%23-descripci%C3%B3n-del-proyecto)
2.  [Dataset](https://www.google.com/search?q=%23-dataset)
3.  [Tecnologías Utilizadas](https://www.google.com/search?q=%23-tecnolog%C3%ADas-utilizadas)
4.  [Estructura del Proyecto y Flujo de Trabajo](https://www.google.com/search?q=%23-estructura-del-proyecto-y-flujo-de-trabajo)
5.  [Cómo Ejecutar el Notebook](https://www.google.com/search?q=%23-c%C3%B3mo-ejecutar-el-notebook)
6.  [Arquitectura del Modelo](https://www.google.com/search?q=%23-arquitectura-del-modelo)
7.  [Resultados (Análisis del Entrenamiento)](https://www.google.com/search?q=%23-resultados-an%C3%A1lisis-del-entrenamiento)
8.  [Autores](https://www.google.com/search?q=%23-autores)

## 🎯 Descripción del Proyecto

El objetivo principal de este proyecto es desarrollar un sistema de visión artificial capaz de:

1.  **Clasificar imágenes de alimentos** en una de las 101 categorías disponibles en el dataset Food-41.
2.  **Realizar una clasificación binaria** sobre la categoría identificada para determinar si el plato es **saludable** o **no saludable**.
3.  Implementar una **Red Neuronal Convolucional (CNN) de 10 capas** con técnicas de regularización como **Dropout** para prevenir el sobreajuste.

Este proyecto sigue las directrices del curso de Aprendizaje de Máquina, aplicando técnicas de preprocesamiento de imágenes, aumento de datos y evaluación de modelos de Deep Learning.

## 🍔 Dataset

Se utiliza el dataset **Food-41**, una versión más manejable del popular Food-101. Este dataset contiene 101,000 imágenes de 101 categorías diferentes de alimentos.

  - **Fuente:** [Food-41 Dataset en Kaggle](https://www.kaggle.com/datasets/kmader/food41/data)
  - **Estructura:** El notebook gestiona la descarga y la organización del dataset en una estructura de carpetas `train/validation/test`, necesaria para el entrenamiento con Keras.

## 🛠️ Tecnologías Utilizadas

  - **Lenguaje:** Python 3.x
  - **Framework de Deep Learning:** TensorFlow y Keras
  - **Librerías Principales:**
      - `Pandas`: Para manipulación de datos inicial.
      - `NumPy`: Para operaciones numéricas.
      - `OpenCV` y `Pillow`: Para procesamiento de imágenes.
      - `Matplotlib` y `Seaborn`: Para visualización de datos y resultados.
      - `scikit-learn`: Para métricas de evaluación adicionales.
      - `tqdm`: Para barras de progreso visuales.
  - **Entorno de Ejecución:** Google Colab (con GPU).

## 📂 Estructura del Proyecto y Flujo de Trabajo

El proyecto se desarrolla en un único notebook de Google Colab y sigue un flujo de trabajo estructurado:

1.  **Configuración del Entorno:** Instalación de librerías y configuración de la API de Kaggle para la descarga del dataset.
2.  **Limpieza de Imágenes:** Se implementa un script que verifica la integridad de todos los archivos de imagen en el dataset, eliminando aquellos que están corruptos y que podrían detener el entrenamiento.
3.  **Organización del Dataset:** Dado que la versión descargada del dataset no incluye archivos de partición (`train.txt`, `test.txt`), se realiza una **división estratificada** manual de los datos:
      - **70%** para el conjunto de **entrenamiento**.
      - **15%** para el conjunto de **validación**.
      - **15%** para el conjunto de **prueba**.
        Los archivos se copian a una nueva estructura de directorios (`dataset_organized/train`, `dataset_organized/validation`, `dataset_organized/test`) compatible con Keras.
4.  **Creación de Generadores de Datos:** Se utiliza `ImageDataGenerator` de Keras para:
      - Cargar imágenes en lotes (batches) desde el disco.
      - Reescalar los valores de los píxeles de `[0, 255]` a `[0, 1]`.
      - Aplicar **aumento de datos (Data Augmentation)** al conjunto de entrenamiento (rotaciones, zooms, volteos, etc.) para mejorar la robustez del modelo.
5.  **Definición del Modelo CNN:** Se construye una CNN secuencial de 10 capas principales con `Dropout` para regularización.
6.  **Entrenamiento del Modelo:** El modelo se entrena utilizando los generadores de datos. Se emplean callbacks como `ModelCheckpoint` (para guardar el mejor modelo) y `EarlyStopping` (para detener el entrenamiento si no hay mejora).
7.  **Visualización y Evaluación:** Se grafican las curvas de precisión y pérdida del entrenamiento y se evalúa el modelo final sobre el conjunto de prueba.
8.  **Clasificación Saludable/No Saludable:** Se implementa la lógica final para clasificar la categoría predicha.

## 🚀 Cómo Ejecutar el Notebook

Para ejecutar este proyecto en Google Colab, sigue estos pasos:

1.  **Abrir en Google Colab:** Abre el archivo `.ipynb` en Google Colab.
2.  **Configurar la API de Kaggle:**
      - Ve a tu perfil de Kaggle (`https://www.kaggle.com/`), entra en "Account" y en la sección "API", haz clic en "Create New API Token" para descargar tu archivo `kaggle.json`.
      - Sube este archivo `kaggle.json` a la sesión de Colab usando el panel de "Archivos" a la izquierda.
3.  **Ejecutar las Celdas en Orden:**
      - **Celda 1-3:** Ejecútalas para instalar dependencias, configurar las credenciales y descargar/extraer el dataset. Este paso puede tardar varios minutos.
      - **Celda 4-5:** Ejecútalas para limpiar y organizar el dataset en las carpetas `train/validation/test`. Este paso también es largo debido a la copia de archivos.
      - **Celda 6-7:** Crean los generadores de datos y visualizan un ejemplo de aumento de datos.
      - **Celda 8:** Define la arquitectura del modelo CNN.
      - **Celda 9:** **Inicia el entrenamiento**. Este es el paso más largo y puede durar varias horas.
      - **Celdas Siguientes:** Ejecútalas para visualizar los resultados del entrenamiento y evaluar el modelo.

## 🧠 Arquitectura del Modelo

El modelo es una Red Neuronal Convolucional secuencial con 10 capas principales, diseñada para la clasificación de imágenes. La arquitectura es la siguiente:

```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 128, 128, 32)      896       
                                                                 
 max_pooling2d (MaxPooling2  (None, 64, 64, 32)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 64, 64, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 32, 32, 64)        0         
 g2D)                                                            
                                                                 
 dropout (Dropout)           (None, 32, 32, 64)        0         
                                                                 
 conv2d_2 (Conv2D)           (None, 32, 32, 128)       73856     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 16, 16, 128)       0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 32768)             0         
                                                                 
 dense (Dense)               (None, 512)               16777728  
                                                                 
 dropout_1 (Dropout)         (None, 512)               0         
                                                                 
 dense_1 (Dense)             (None, 101)               51813     
                                                                 
=================================================================
Total params: 16,922,789
Trainable params: 16,922,789
Non-trainable params: 0
_________________________________________________________________
```

## 📊 Resultados (Análisis del Entrenamiento)

A continuación se muestran las curvas de aprendizaje del modelo después del entrenamiento.

#### Curvas de Precisión y Pérdida

*[Inserta aquí la imagen de los gráficos de `accuracy` y `loss` vs. épocas que se genera al final del entrenamiento. Esto mostrará cómo el modelo aprendió y si hubo sobreajuste.]*

#### Métricas de Evaluación Final

*[Inserta aquí una tabla o un resumen con las métricas finales obtenidas en el conjunto de prueba, como Accuracy, Precision, Recall y F1-Score.]*

| Métrica   | Puntuación |
| :-------- | :--------- |
| Accuracy  | XX.XX%     |
| Precision | XX.XX%     |
| Recall    | XX.XX%     |
| F1-Score  | XX.XX%     |

## 👥 Autores

  - **Akhan Lorenzo Andrés Espinoza Rojas**
  - **Roberto López Lizana**
  - **Mariano Mendez Fernandez**

-----