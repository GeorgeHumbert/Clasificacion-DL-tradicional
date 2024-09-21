# Clasificación de Reseñas de Películas IMDb con Deep Learning

Este proyecto utiliza un modelo de Deep Learning basado en LSTM para clasificar reseñas de películas en IMDb como "positivas" o "negativas". El modelo está entrenado usando Keras y TensorFlow, y se implementa una API con Flask para realizar inferencias en tiempo real.

## Estructura del Proyecto

El proyecto está dividido en varias fases y archivos:

- **A_Feature_Pipeline.ipynb**: Preprocesamiento de los datos, incluyendo la tokenización y generación de secuencias para el entrenamiento del modelo.
- **B_Training_Pipeline.ipynb**: Entrenamiento del modelo de Deep Learning (LSTM) con los datos preprocesados.
- **C_Model_Inference.ipynb**: Evaluación del modelo entrenado y prueba con nuevos ejemplos para obtener predicciones.
- **API Fast**: Carpeta que contiene la implementación de la API usando Flask para realizar predicciones en tiempo real. El archivo principal es `app.py`.
- **modelo_imdb_dl.h5**: Archivo que contiene el modelo entrenado guardado en formato HDF5.
- **tokenizer.pkl**: Archivo que contiene el tokenizador utilizado durante el preprocesamiento para transformar el texto en secuencias.

## Requisitos

Para ejecutar este proyecto, necesitas instalar los siguientes paquetes:

```bash
pip install -r requirements.txt
El archivo requirements.txt contiene:

makefile
Copiar código
Flask==2.0.1
joblib==1.1.0
tensorflow==2.6.0
keras==2.6.0
numpy==1.19.5
Instrucciones de Ejecución
1. Entrenamiento del Modelo
Para entrenar el modelo, sigue estos pasos:

Ejecuta el notebook A_Feature_Pipeline.ipynb para preparar los datos.
Luego, ejecuta B_Training_Pipeline.ipynb para entrenar el modelo LSTM.
Guarda el modelo entrenado como modelo_imdb_dl.h5 y el tokenizador como tokenizer.pkl.
2. Implementación de la API
Una vez que hayas entrenado y guardado el modelo, puedes implementar la API para realizar predicciones en tiempo real.

Ve a la carpeta api_fast.
Asegúrate de que los archivos modelo_imdb_dl.h5 y tokenizer.pkl estén en el mismo directorio que el archivo app.py.
Ejecuta el siguiente comando para levantar el servidor Flask:
bash
Copiar código
python app.py
3. Realizar Predicciones
Con el servidor Flask en ejecución, puedes hacer solicitudes a la API para obtener predicciones. Aquí hay un ejemplo de cómo realizar una solicitud POST con curl:

bash
Copiar código
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"review\": \"This movie was fantastic and had great acting!\"}"
La API devolverá una predicción en formato JSON indicando si la reseña es "positiva" o "negativa".

Ejemplos de Uso
Reseña Positiva:

json
Copiar código
{
  "sentiment": "positive"
}
Reseña Negativa:

json
Copiar código
{
  "sentiment": "negative"
}
Resultados
Precisión de Entrenamiento: 98.14%
Precisión de Validación: 87.33%
Pérdida de Validación: 0.4702
Notas Finales
Este proyecto es un ejemplo básico de cómo usar redes neuronales recurrentes (LSTM) para la clasificación de texto.
Se puede mejorar la precisión utilizando técnicas más avanzadas, como el uso de modelos preentrenados como BERT o GPT.
La API implementada puede ser fácilmente desplegada en servicios de nube como Heroku, AWS o Google Cloud.
