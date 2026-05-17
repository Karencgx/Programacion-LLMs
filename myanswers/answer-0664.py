import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluar_regresion_ridge(X_train, y_train, X_test, y_test, alpha):
    
    # 1. Crear modelo Ridge
    modelo = Ridge(alpha=alpha)
    
    # 2. Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # 3. Generar predicciones
    y_pred = modelo.predict(X_test)
    
    # 4. Calcular métricas
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    
    rmse = round(
        float(np.sqrt(mean_squared_error(y_test, y_pred))),
        4
    )
    
    r2 = round(r2_score(y_test, y_pred), 4)
    
    # 5. Retornar diccionario
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }