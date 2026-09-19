# Carga de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga de datos
df = pd.read_csv("smart_workout_raw_dataset.csv")
# Dimensiones del DataFrame
df.shape
# Impresión de las primeras filas
df.head()
# Identificación del tipo de cada variable
df.info()