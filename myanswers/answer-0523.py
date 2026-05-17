import numpy as np
import pandas as pd

def calcular_mse_filtrado(df):
    
    # 1. Filtrar filas donde reales > 0
    df_filtrado = df[df["reales"] > 0]
    
    # 2. Calcular diferencias al cuadrado
    errores_cuadrados = (
        df_filtrado["reales"] - df_filtrado["predicciones"]
    ) ** 2
    
    # 3. Calcular promedio (MSE)
    mse = float(np.mean(errores_cuadrados))
    
    return mse