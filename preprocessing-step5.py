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
