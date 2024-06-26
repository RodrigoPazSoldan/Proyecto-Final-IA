import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re



def expand_contractions(text):
    # Definimos un diccionario con las contracciones comunes y sus formas expandidas
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