# Eliot Destaing

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import datetime

app = Flask(__name__)

# Configuración de la base de datos
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo de datos para las reseñas
class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    review_text = db.Column(db.String(500), nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)
    date_posted = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Crear las tablas de la base de datos
db.create_all()

# Cargar el modelo y el tokenizer
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    # Preprocesamiento del texto
    seq = tokenizer.texts_to_sequences([review])
    padded_seq = pad_sequences(seq, maxlen=100)  # Asegúrate de que la longitud máxima sea la utilizada durante el entrenamiento
    prediction = model.predict(padded_seq)
    sentiment = np.argmax(prediction)
    sentiment_label = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
    predicted_sentiment = sentiment_label[sentiment]

    # Guardar la predicción en la base de datos
    new_review = Review(review_text=review, sentiment=predicted_sentiment)
    db.session.add(new_review)
    db.session.commit()

    return render_template('result.html', sentiment=predicted_sentiment, review=review)

@app.route('/history')
def history():
    # Recuperar todas las reseñas de la base de datos
    reviews = Review.query.all()
    return render_template('history.html', reviews=reviews)

if __name__ == '__main__':
    app.run(debug=True)
