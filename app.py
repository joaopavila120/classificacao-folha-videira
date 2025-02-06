import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import image
import numpy as np
import pandas as pd
import plotly.express as px

def carrega_modelo():
    #url = 'https://colab.research.google.com/drive/1b7Da-d8xekZfU6E62FSZNFoUOZCp06gK?usp=drive_link'
    url = 'https://drive.google.com/uc?id=1b7Da-d8xekZfU6E62FSZNFoUOZCp06gK'

    gdown.download(url, 'modelo_quantizado16bits.tflite')
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file.uploader('Arraste ou solte uma imagem', type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))

        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image
    
def previsao(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['BlackMasles', 'BlackRot', 'HealthyGrapes', 'LeafBlight']

    df = pf.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Doenças em Uvas')

    st.plotly_chart(fig)

def main(): 
    st.set_page_config(
        page_title = "Classificação de folhas de videira",
        page_icon = " 🍇 "
    )

    st.write("# Classificação de folhas de videira!  🍇 ")

    interpreter = carrega_modelo()
    image = carrega_imagem()

    if image is not None:
        previsao(interpreter, image)


if __name__ == "__main__":
    main()