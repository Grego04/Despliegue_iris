
import streamlit as st
import pandas as pd
import joblib
import os

# Change to the directory where the models are located
# This path should match the `os.chdir` from previous cells
# Assuming '/content/drive/MyDrive/Ejercicios análisis de datos IA/Despliegue iris ' is the correct path
MODEL_PATH = '/content/drive/MyDrive/Ejercicios análisis de datos IA/Despliegue iris '
os.chdir(MODEL_PATH)

st.set_page_config(page_title='Iris Species Prediction', layout='centered')

@st.cache_resource
def load_models():
    try:
        best_knn_model = joblib.load('best_knn_model.joblib')
        label_encoder_species = joblib.load('label_encoder_species.joblib')
        return best_knn_model, label_encoder_species
    except FileNotFoundError:
        st.error(f"Error: Model files not found in {MODEL_PATH}. Please ensure 'best_knn_model.joblib' and 'label_encoder_species.joblib' are in the correct directory.")
        st.stop()

best_knn_model, label_encoder_species = load_models()

st.title('Iris Flower Species Prediction')
st.write('Enter the measurements of the Iris flower to predict its species.')

# Input features
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.3, 0.1)
peta_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2, 0.1)

# Prepare data for prediction
# Feature names must match the ones used during training
feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
new_flower_data = pd.DataFrame([[
    sepal_length,
    sepal_width,
    petal_length,
    peta_width
]], columns=feature_names)

if st.button('Predict Species'):
    predicted_species_encoded = best_knn_model.predict(new_flower_data)
    predicted_species = label_encoder_species.inverse_transform(predicted_species_encoded)
    st.success(f'The predicted Iris species is: **{predicted_species[0]}**')

st.write("""
--- 
This app predicts the species of an Iris flower based on its sepal and petal measurements.
""")
