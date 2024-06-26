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
