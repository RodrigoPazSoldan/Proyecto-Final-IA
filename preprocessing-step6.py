sentimient_mapping = {'positive': 1, 'negative': 2, 'neutral': 0}

# Aplicar el mapeo a la columna 'sentimiento'
df_balanced['Sentiment_labeled'] = df_balanced['Sentiment'].map(sentimient_mapping)
