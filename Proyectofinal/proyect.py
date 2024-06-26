##F1

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
import pickle


def expand_contractions(text):

    contractions_dict = {
        "ain't": "is not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "to've": "to have",
        "wasn't": "was not",
        "weren't": "were not",
        "what'll": "what will",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
    }

    contractions_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b')

    # Expand function
    def expand_match(contraction):
        return contractions_dict[contraction.group(0)]

    # Verify if text is str
    if isinstance(text, str):

        expanded_text = contractions_pattern.sub(expand_match, text)
    else:

        expanded_text = text

    return expanded_text



##F2

#Import Dataset

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/sentiment.csv', encoding='latin-1')

count_positivo = 0
count_negativo = 0
count_neutral = 0

for index, row in df.iterrows():
    if row['Sentiment'] == 'positive':
        count_positivo += 1
    elif row['Sentiment'] == 'negative':
        count_negativo += 1
    elif row['Sentiment'] == 'neutral':
        count_neutral += 1

# Number of rows for each category
num_per_category = 5000

df_positive = df[df['Sentiment'] == 'positive'].sample(n=num_per_category, replace=True)
df_negative = df[df['Sentiment'] == 'negative'].sample(n=num_per_category, replace=True)
df_neutral = df[df['Sentiment'] == 'neutral'].sample(n=num_per_category, replace=True)

df_balanced = pd.concat([df_positive, df_negative, df_neutral])






##  F3

def eliminar_filas_nan(df, columna):

    df_sin_nan = df.dropna(subset=[columna])

    return df_sin_nan

df_sin_nan = eliminar_filas_nan(df_balanced, 'Summary')

# Convert df to a list
Reviews = df_sin_nan['Summary'].tolist()



## F4

nltk.download('stopwords')
nltk.download('punkt')

import string
import re

def remove_in_words(reviews):
    punct = set(string.punctuation)
    text = []

    for paragraph in reviews:
        cleaned_paragraph = []
        for word in paragraph.split():
            cleaned_chars = []
            previous_char = None
            for letra in word:
                if letra.isdigit():
                    continue
                if letra in punct:
                    if letra != previous_char:
                        cleaned_chars.append(letra)
                    previous_char = letra
                else:
                    cleaned_chars.append(letra)
                    previous_char = letra

            cleaned_word = ''.join(cleaned_chars)
            cleaned_paragraph.append(cleaned_word)

        cleaned_paragraph = ' '.join(cleaned_paragraph)
        text.append(cleaned_paragraph)

    return text

cleaned_reviews = remove_in_words(Reviews)



## F5


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Delete stopwords
def pre_process(reviews):
    stopset = set(stopwords.words('english'))
    # Words that does not have to delete
    whitelist = {'not', 'no', 'nor'}

    processed_reviews = []
    for corp in reviews:
        corpus = corp.lower()
        words = word_tokenize(corpus)
        filtered_words = []
        for word in words:
            if word in stopset and word not in whitelist:
                continue
            filtered_words.append(word)
        processed_review = " ".join(filtered_words)
        processed_reviews.append(processed_review)

    return processed_reviews

df2 = pre_process(cleaned_reviews)



##F6

sentimiento_mapping = {'positive': 1, 'negative': 2, 'neutral': 0}

# Aplicar el mapeo a la columna 'sentimiento'
df_balanced['Sentiment_labeled'] = df_balanced['Sentiment'].map(sentimiento_mapping)
print(df_balanced['Sentiment_labeled'])

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Labels to categorical
y = df_balanced['Sentiment_labeled']

y = to_categorical(y, num_classes=3)

# Sequences and token
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df2)
sequences = tokenizer.texts_to_sequences(df2)
X = pad_sequences(sequences, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Max lenght of the sequences
max_length = X.shape[1]

# Model
model = Sequential()
model.add(LSTM(100, input_shape=(max_length, 1), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(100))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Adjust the shape of data for LSTM
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Training
history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test), batch_size=32)

# Evaluation
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Guardar el modelo
model.save('sentiment_model.h5')  # Guardar como .h5 para compatibilidad

# Guardar el tokenizer usando pickle
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)