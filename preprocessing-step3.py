def eliminar_filas_nan(df, columna):

    df_sin_nan = df.dropna(subset=[columna])

    return df_sin_nan

df_sin_nan = eliminar_filas_nan(df_balanced, 'Summary')

# Convert df to a list
Reviews = df_sin_nan['Summary'].tolist()
