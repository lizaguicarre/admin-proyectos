import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Librerías: pandas, sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Carga de datos
data = pd.read_excel('C:/Users/usuario/Downloads/Tamborrel Data (4)/Tamborrel Data/Casas Vendidas The Woodlands/ventas 2022.xlsx')

# División entre test (30%) y train (70%)
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)

# Variables independientes
input_cols = ['building_sqft', 'bedrooms', 'baths_total']
# Variable dependiente
output_cols = ['close_price']

train_inputs = train_data[input_cols]
train_outputs = train_data[output_cols]

test_inputs = test_data[input_cols]
test_outputs = test_data[output_cols]

# ------------------------------------------------------------------------------------------------------------
# Modelo 1
# ------------------------------------------------------------------------------------------------------------
# Modelo 2
# ------------------------------------------------------------------------------------------------------------
# Modelo 3
# ------------------------------------------------------------------------------------------------------------
# Random Forest model
rf_model = RandomForestRegressor(random_state=42)
# Definir un grid para hyperparameter tunning
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [5, 10, 15, None]}

# Buscar los mejores parámetros con ayuda del grid, 5 Cross-Validation Folds
grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_inputs, train_outputs)

# Entrenar el modelo con los mejores parámetros
best_rf_model = grid_search.best_estimator_
best_rf_model.fit(train_inputs, train_outputs)

# Evaluación del modelo (Accuracy)
rf_score = best_rf_model.score(test_inputs, test_outputs)
# -----------------------------------------------------------------------------------------------------------

# Predicciones
print("Predicción del precio de casas")
print("Ingresa los siguientes datos de la propiedad que deseas vender:")

sqft = float(input("Metros cuadrados: "))
num_rooms = int(input("Número de cuartos: "))
num_bathrooms = float(input("Número de baños completos: "))

#Input prueba

input_data = [[num_rooms, sqft, num_bathrooms]]

# Modelo 1 - Predicciones
# Modelo 2 - Predicciones
# Modelo 3 - Predicciones
# Random Forest - Predicciones
rf_price = best_rf_model.predict(input_data)

print("------------------------------------------------------------------------------------------------------")
print("Resultados:")
# print("Precio estimado con modelo 1: ${:.2f}, con precisión del: {:.2f}%".format(modelo1_price[0], modelo1_score * 100))
# print("Precio estimado con modelo 2: ${:.2f}, con precisión del: {:.2f}%".format(modelo2_price[0], modelo2_score * 100))
# print("Precio estimado con modelo 3: ${:.2f}, con precisión del: {:.2f}%".format(modelo3_price[0], modelo3_score * 100))
print("Precio estimado con random forest: ${:.2f}, con precisión del: {:.2f}%".format(rf_price[0], rf_score * 100))
