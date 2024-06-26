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
