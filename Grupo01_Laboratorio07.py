# Carga de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carga de datos
df = pd.read_csv("smart_workout_raw_dataset.csv")
# Dimensiones del DataFrame
print(df.shape)
# Impresión de las primeras filas
print(df.head())
# Identificación del tipo de cada variable
df.info()

# Creando nuevas variables con valores en minúsculas y sin espacios vacíos
categorical_columns = [
 "persona",
 "fitness_goal",
 "experience_level",
 "injury_constraint",
 "bodyPart",
 "equipment",
 "target_muscle"
]
for column in categorical_columns:
    df[column + "_std"] = (df[column].astype("string").str.strip().str.lower().str.replace(" ", "_"))
 
print(df[["fitness_goal", "fitness_goal_std"]].drop_duplicates().sort_values("fitness_goal_std"))

df_usuario = (df.groupby("user_id").agg(
    age = ("age", "median"),
    n_records = ("user_id", "size"),
    n_exercises = ("exercise_id", "nunique"),
    n_bodyparts = ("bodyPart_std", "nunique"),
    n_equipment = ("equipment_std", "nunique"),
    n_target_muscles = ("target_muscle_std", "nunique"),
    mean_rating = ("rating", "mean"),
    median_rating = ("rating", "median"),
    sd_rating = ("rating", "std")
    ).reset_index()
              )
# Dimensiones del nuevo DataFrame
print(df_usuario.shape)

# Impresión de las primeras filas
print(df_usuario.head())

# Calculando el conteo de objetivos por usuario
conteo_objetivos = pd.crosstab(
 df["user_id"],
 df["fitness_goal_std"]
)

# Añadiendo al nombre de cada variable el prefijo n_goal_
conteo_objetivos = conteo_objetivos.add_prefix("n_goal_")

# Calculando el conteo de partes del cuerpo ejercitadas por usuario
conteo_bodyparts = pd.crosstab(
 df["user_id"],
 df["bodyPart_std"]
)

# Añadiendo al nombre de cada variable el prefijo n_body_
conteo_bodyparts = conteo_bodyparts.add_prefix("n_body_")

# Agregando variables que cuentan objetivos
df_usuario = df_usuario.merge(
 conteo_objetivos,
 on = "user_id",
 how = "left"
)

# Agregando variables que cuentan partes del cuerpo ejercitadas
df_usuario = df_usuario.merge(
 conteo_bodyparts,
 on="user_id",
 how="left"
)

# Proporción de cantidad de registros en general fitness
df_usuario["share_general_fitness"] = (
 df_usuario["n_goal_general_fitness"]
 / df_usuario["n_records"]
)

# Proporción de cantidad de registros en endurance
df_usuario["share_endurance"] = (
 df_usuario["n_goal_endurance"]
 / df_usuario["n_records"]
)

print(df_usuario["user_id"].nunique())
print(df_usuario.shape[0])
print(df_usuario["user_id"].duplicated().sum())
print(df_usuario.info())

# Punto 2
# Par 1: 
r1 = df_usuario["n_goal_general_fitness"].corr(df_usuario["mean_rating"])
plt.figure()
plt.scatter(df_usuario["n_goal_general_fitness"], df_usuario["mean_rating"])
plt.xlabel("Número de registros con objetivo general fitness")
plt.ylabel("Rating promedio del usuario")
plt.title("Objetivo general fitness vs Rating (r = " + str(round(r1, 3)) + ")")
plt.show()
print("Correlación n_goal_general_fitness & mean_rating:", r1)

# Par 2:
r2 = df_usuario["n_bodyparts"].corr(df_usuario["mean_rating"])
plt.figure()
plt.scatter(df_usuario["n_bodyparts"], df_usuario["mean_rating"])
plt.xlabel("Número de partes del cuerpo distintas")
plt.ylabel("Rating promedio del usuario")
plt.title("Diversidad del entrenamiento vs Rating (r = " + str(round(r2, 3)) + ")")
plt.show()
print("Correlación n_bodyparts & mean_rating:", r2)

# Par 3:
r3 = df_usuario["age"].corr(df_usuario["n_bodyparts"])
plt.figure()
plt.scatter(df_usuario["age"], df_usuario["n_bodyparts"])
plt.xlabel("Edad del usuario (años)")
plt.ylabel("Número de partes del cuerpo distintas")
plt.title("Edad vs diversidad del entrenamiento (r = " + str(round(r3, 3)) + ")")
plt.show()
print("Correlación age & n_bodyparts:", r3)

# PUNTO 3

#Se crea el diagrama de caja y bigotes para la variable n_bodyparts.
df_usuario["n_bodyparts"].plot(kind="box")
plt.show()

# Se crea un dataframe con solo 2 columnas, correspondientes a user_id y n_bodyparts.
df_usuario_n_bodyparts = df_usuario[['user_id', 'n_bodyparts']]

# Se ordenan los valores según la columna n_bodyparts para identificar los usuarios
# correspondientes a los valores atípicos superiores e inferiores.
n_bodyparts_ordenado = df_usuario_n_bodyparts.sort_values(by='n_bodyparts', ascending=True)
print(n_bodyparts_ordenado.head(10))
print(n_bodyparts_ordenado.tail())

#Se crea el diagrama de caja y bigotes para la variable mean_rating.
df_usuario["mean_rating"].plot(kind="box")
plt.show()

# Se crea un dataframe con solo 2 columnas, correspondientes a user_id y mean_rating.
df_usuario_mean_rating = df_usuario[['user_id', 'mean_rating']]

# Se ordenan los valores según la columna mean_rating para identificar los usuarios
# correspondientes a los valores atípicos inferiores.
mean_rating_ordenado = df_usuario_mean_rating.sort_values(by='mean_rating', ascending=True)
print(mean_rating_ordenado.head())

# PUNTO 4

# Coeficiente de correlación de Pearson entre:
# mean_rating y n_goal_general_fitness
print(df_usuario["mean_rating"].corr(df_usuario["n_goal_general_fitness"]))
# mean_rating y n_equipment
print(df_usuario["mean_rating"].corr(df_usuario["n_equipment"]))
# n_equipment y n_goal_general_fitness
print(df_usuario["n_equipment"].corr(df_usuario["n_goal_general_fitness"]))

# PUNTO 5

# Seleccionamos endurance como tercera variable.
variables = df_usuario[[
    "share_general_fitness",
    "mean_rating",
    "share_endurance"
]].dropna()

# Calculamos las correlaciones de Pearson.
correlaciones = variables.corr()

r_xy = correlaciones.loc["share_general_fitness", "mean_rating"]
r_xz = correlaciones.loc["share_general_fitness", "share_endurance"]
r_yz = correlaciones.loc["mean_rating", "share_endurance"]

# Calculamos la correlacion parcial controlando endurance.
r_parcial = (r_xy - r_xz * r_yz) / np.sqrt(
    (1 - r_xz**2) * (1 - r_yz**2)
)

# Mostramos los resultados en un solo print.
print(
    "PUNTO 5\n",
    "Correlacion original:", round(r_xy, 3),
    "\nCorrelacion general_fitness y endurance:", round(r_xz, 3),
    "\nCorrelacion endurance y rating:", round(r_yz, 3),
    "\nCorrelacion parcial:", round(r_parcial, 3)
)