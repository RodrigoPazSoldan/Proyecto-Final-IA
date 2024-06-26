from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Charger le modèle et le tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    # Preprocessing du texte
    seq = tokenizer.texts_to_sequences([review])
    padded_seq = pad_sequences(seq, maxlen=100)  # Assurez-vous que la longueur maximale correspond à celle utilisée lors de l'entraînement
    prediction = model.predict(padded_seq)
    # Interpréter la prédiction
    sentiment = np.argmax(prediction)  # Obtenir l'indice de la classe prédite
    sentiment_label = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    predicted_sentiment = sentiment_label[sentiment]
    return render_template('result.html', sentiment=predicted_sentiment, review=review)

@app.route('/history')
def history():
    # Vous pouvez connecter cette route à une base de données ou un fichier pour montrer l'historique
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True)
