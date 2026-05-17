import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA

def calcular_correlacion_canonica(df: pd.DataFrame,
                                  habit_cols: list[str],
                                  result_cols: list[str]) -> pd.DataFrame:
    
    # 1. Seleccionar columnas y eliminar filas con nulos
    selected_df = df[habit_cols + result_cols].copy()
    cleaned_df = selected_df.dropna(axis=0, how="any").copy()
    
    # 2. Convertir a numpy arrays float
    X = cleaned_df[habit_cols].to_numpy(dtype=float)
    Y = cleaned_df[result_cols].to_numpy(dtype=float)
    
    # 3. Ajustar CCA y transformar
    cca_model = CCA(n_components=1)
    cca_model.fit(X, Y)
    
    X_c, Y_c = cca_model.transform(X, Y)
    
    # Correlación de Pearson
    canonical_correlation = float(
        np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
    )
    
    # 4. Retornar DataFrame
    resultado = pd.DataFrame({
        "correlacion_canonica": [canonical_correlation],
        "n_muestras_utilizadas": [int(len(cleaned_df))]
    })
    
    return resultado