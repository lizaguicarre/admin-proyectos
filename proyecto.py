import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Librerías: pandas, sklearn, tensorflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

# Carga de datos
data = pd.read_excel('/Users/liz/Downloads/SEXTO SEMESTRE/CONCENTRACIÓN - INTELIGENCIA ARTIFICIAL CON IMPACTO EMPRESARIAL/RETO/DATOS RETO/ventas 2022.xlsx')

# División entre test (30%) y train (70%)
train_data = data.sample(frac=0.7, random_state=42)
test_data = data.drop(train_data.index)
## Divisiòn para redes neuronales
x_columns = data.columns.drop('close_price')
x = data[x_columns].values
y = data['close_price'].values
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.30, random_state=42)

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
nn_model.add(Dense(25, input_dim=x.shape[1], activation='relu')) # Hidden 1
nn_model.add(Dense(10, activation='relu')) # Hidden 2
nn_model.add(Dense(1)) # Output
nn_model.compile(loss='mean_squared_error', optimizer='adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
                        patience=5, verbose=1, mode='auto', 
                        restore_best_weights=True)

# Prueba del modelo
nn_model.fit(x_train,y_train,validation_data=(x_test,y_test),
          callbacks=[monitor],verbose=2,epochs=1000)

# Evaluación del modelo
nn_score = nn_model.score(test_inputs, test_outputs)

# Predecir con el modelo
nn_price = nn_model.predict(input_data)
#nn_rmse = metrics.mean_squared_error(pred,y_test)
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

# Evaluación del modelo
rf_score = best_rf_model.score(test_inputs, test_outputs)
# -----------------------------------------------------------------------------------------------------------

# Predicciones
print("Predicción del precio de casas")
print("Ingresa los siguientes datos de la propiedad que deseas vender:")

sqft = float(input("Metros cuadrados: "))
num_rooms = int(input("Número de cuartos: "))
num_bathrooms = float(input("Número de baños completos: "))

input_data = [[num_rooms, sqft, num_bathrooms]]

# Modelo 1 - Predicciones
# Modelo 2 - Predicciones
# Modelo 3 - Predicciones
# Random Forest - Predicciones
rf_price = best_rf_model.predict(input_data)

print("------------------------------------------------------------------------------------------------------")
print("Resultados:")
print("Precio estimado con redes neuronales: ${:.2f}, con precisión del: {:.2f}%".format(nn_price[0], nn_score * 100))
# print("Precio estimado con modelo 2: ${:.2f}, con precisión del: {:.2f}%".format(modelo2_price[0], modelo2_score * 100))
# print("Precio estimado con modelo 3: ${:.2f}, con precisión del: {:.2f}%".format(modelo3_price[0], modelo3_score * 100))
print("Precio estimado con random forest: ${:.2f}, con precisión del: {:.2f}%".format(rf_price[0], rf_score * 100))