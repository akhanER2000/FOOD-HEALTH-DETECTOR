import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Clasificador de Comida Saludable",
    page_icon="🥗",
    layout="centered"
)

# --- Carga del Modelo ---
# Usamos el decorador de caché de Streamlit para que el modelo se cargue solo una vez.
@st.cache_resource
def load_food_model():
    try:
        model = tf.keras.models.load_model('foodnet_model.h5')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

model = load_food_model()

# --- Funciones de Preprocesamiento y Predicción ---
def preprocess_image(image):
    """Preprocesa la imagen para que coincida con la entrada del modelo."""
    img = image.resize((64, 64))
    img_array = np.array(img)
    
    # Si la imagen es blanco y negro o tiene canal alfa, la convierte a RGB
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_array, class_names):
    """Realiza una predicción y devuelve la clase y la confianza."""
    prediction = model.predict(image_array)[0][0]
    if prediction > 0.5:
        # El índice 1 suele ser 'saludable' si el generador los ordena alfabéticamente
        predicted_class = class_names[1]
        confidence = prediction
    else:
        # El índice 0 es 'no_saludable'
        predicted_class = class_names[0]
        confidence = 1 - prediction
    return predicted_class, confidence


# --- Interfaz de Usuario ---
st.title("🥗 Detector de Comida Saludable 🍕")
st.write(
    "Sube una imagen de un plato de comida y la inteligencia artificial "
    "clasificará si es 'saludable' o 'no saludable'."
)

uploaded_file = st.file_uploader(
    "Elige una imagen...", type=["jpg", "jpeg", "png"]
)

if model is None:
    st.warning("El modelo no está cargado. Asegúrate de que 'foodnet_model.h5' existe.")
else:
    if uploaded_file is not None:
        # Mostrar la imagen
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada.", use_column_width=True)
        
        # Clasificar la imagen al presionar el botón
        if st.button("Clasificar Comida"):
            with st.spinner('Analizando la imagen...'):
                # Las clases deben coincidir con las del entrenamiento
                class_names = ['no_saludable', 'saludable']
                
                # Preprocesar y predecir
                preprocessed_array = preprocess_image(image)
                predicted_class, confidence = predict(preprocessed_array, class_names)
                
                # Mostrar resultado
                if predicted_class == "saludable":
                    st.success(f"**Resultado: ¡Saludable!** (Confianza: {confidence:.2%})")
                else:
                    st.error(f"**Resultado: ¡No Saludable!** (Confianza: {confidence:.2%})")