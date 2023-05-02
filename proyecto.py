import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# Librerías: pandas, sklearn
import pandas as pd

# Carga de datos
data = pd.read_excel('path/ventas 2022.xlsx')

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
# Modelo 4
# -----------------------------------------------------------------------------------------------------------

# Predicciones
print("Predicción del precio de casas")
num_rooms = int(input("Número de cuartos: "))
sqft = float(input("Metros cuadrados: "))
num_bathrooms = float(input("Número de baños completos: "))

input_data = [[num_rooms, sqft, num_bathrooms]]

# Modelo 1 - Predicciones
# Modelo 2 - Predicciones
# Modelo 3 - Predicciones
# Modelo 4 - Predicciones

print("------------------------------------------------------------------------------------------------------")
print("Resultados:")
# print("Precio estimado con modelo 1: ${:.2f}, con precisión del: {:.2f}%".format(modelo1_price[0], modelo1_score * 100))
# print("Precio estimado con modelo 2: ${:.2f}, con precisión del: {:.2f}%".format(modelo2_price[0], modelo2_score * 100))
# print("Precio estimado con modelo 3: ${:.2f}, con precisión del: {:.2f}%".format(modelo3_price[0], modelo3_score * 100))
# print("Precio estimado con modelo 4: ${:.2f}, con precisión del: {:.2f}%".format(modelo4_price[0], modelo4_score * 100))