import pandas as pd

#Import Dataset

df = pd.read_csv('/content/sentiment (1).csv', encoding='latin-1')

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
