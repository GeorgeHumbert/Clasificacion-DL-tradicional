from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

app = Flask(__name__)

# Cargar el modelo entrenado y el tokenizador
model = tf.keras.models.load_model('modelo_imdb_dl.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del request
        data = request.get_json(force=True)
        review_text = data['review']
        
        # Tokenizar y padear el texto
        sequences = tokenizer.texts_to_sequences([review_text])
        if len(sequences[0]) == 0:
            return jsonify({"error": "Error al tokenizar el texto. Verifica que el tokenizador esté correctamente cargado."}), 400
        
        padded = pad_sequences(sequences, maxlen=200)
        
        # Hacer la predicción
        prediction = model.predict(padded)
        
        # Interpretar la predicción
        sentiment = "positive" if prediction[0] > 0.5 else "negative"
        
        # Devolver el resultado como JSON
        return jsonify({"review": review_text, "sentiment": sentiment})
    
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error al procesar la solicitud: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

