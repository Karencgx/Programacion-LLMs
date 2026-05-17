import pandas as pd
import numpy as np

def encode_frecuencia(df):
    
    # calcular frecuencia relativa de cada ciudad
    counts = df["ciudad"].value_counts(normalize=True)
    
    # mapear frecuencia a cada fila
    resultado = df["ciudad"].map(counts).values
    
    return resultado