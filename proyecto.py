import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Librerías: pandas, sklearn, tensorflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Carga de datos
data = pd.read_excel('C:/Users/usuario/Downloads/Tamborrel Data (4)/Tamborrel Data/Casas Vendidas The Woodlands/ventas 2022.xlsx')

# División entre test (30%) y train (70%)
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)
## Divisiòn para redes neuronales
#x_columns = data.columns.drop('close_price')
#x = data[x_columns].values
#y = data['close_price'].values
#x_train, x_test, y_train, y_test = train_test_split(    
#    x, y, test_size=0.30, random_state=42)

# Variables independientes
input_cols = ['building_sqft', 'bedrooms', 'baths_total']
# Variable dependiente
output_cols = ['close_price']

train_inputs = train_data[input_cols]
train_outputs = train_data[output_cols]

test_inputs = test_data[input_cols]
test_outputs = test_data[output_cols]

# ------------------------------------------------------------------------------------------------------------
# Redes Neuronales
# Especificación del modelo
nn_model = Sequential()
nn_model.add(Dense(25, input_dim=3, activation='relu')) # Hidden 1
nn_model.add(Dense(10, activation='relu')) # Hidden 2
nn_model.add(Dense(1)) # Output
nn_model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto', 
                        restore_best_weights=True)

# Prueba del modelo
nn_model.fit(train_inputs, train_outputs, epochs=50, batch_size=32, validation_split=0.2, callbacks=[monitor])

# Evaluación del modelo
nn_preds = nn_model.predict(test_inputs)
nn_mse = metrics.mean_squared_error(test_outputs, nn_preds)
nn_score = np.sqrt(nn_mse) #RMSE

#nn_rmse = metrics.mean_squared_error(pred,y_test)
# ------------------------------------------------------------------------------------------------------------
# Modelo 2
# ------------------------------------------------------------------------------------------------------------
# Decision Tree Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

#  Modelo
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_split=7, min_samples_leaf=8, max_features=8)

# Entrenando modelo
dt_model.fit(train_inputs, train_outputs)

# Creando prediciones
dt_pred = dt_model.predict(test_inputs)

# Evaluate the model
from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(test_outputs, dt_pred)

rmse = math.sqrt(mean_squared_error(test_outputs, dt_pred))
print("RMSE:", rmse)
print("Accuracy:", accuracy)

#Modelo Joshua
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

# Predicciones iniciales con el modelo
y_pred = best_rf_model.predict(test_inputs)

# Evaluación del modelo
rf_score = rmse = np.sqrt(metrics.mean_squared_error(test_outputs, y_pred))

# -----------------------------------------------------------------------------------------------------------

# Predicciones
print("Predicción del precio de casas")
print("Ingresa los siguientes datos de la propiedad que deseas vender:")

sqft = float(input("Metros cuadrados: "))
num_rooms = int(input("Número de cuartos: "))
num_bathrooms = float(input("Número de baños completos: "))

input_data = [[num_rooms, sqft, num_bathrooms]]

# Neural Network - Predicciones
nn_price = nn_model.predict(input_data)
# Modelo 2 - Predicciones
# Modelo 3 - Predicciones
dt_price = dt_pred.predict(input_data)

# Random Forest - Predicciones
rf_price = best_rf_model.predict(input_data)

print("------------------------------------------------------------------------------------------------------")
print("Resultados:")
print("Precio estimado con redes neuronales: ${:.2f}, con un RMSE del: {:.2f}".format(float(nn_price), nn_score))
# print("Precio estimado con modelo 2: ${:.2f}, con un RMSE del: {:.2f}%".format(modelo2_price[0], modelo2_score * 100))
# print("Precio estimado con modelo 3: ${:.2f}, con un RMSE del: {:.2f}%".format(modelo3_price[0], modelo3_score * 100))
print("Precio estimado con random forest: ${:.2f}, con un RMSE del: {:.2f}".format(rf_price[0], rf_score))