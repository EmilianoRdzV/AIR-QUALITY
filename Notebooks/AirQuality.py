# coding: utf-8


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# [![UAEM](https://www.uaem.mx/fcaei/images/uaem.png)](https://www.uaem.mx/fcaei/moca.html)
# [![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
# [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/EmilianoRdzV/AIR-QUALITY)
# 
# # Proyecto: Análisis y Predicción de la Calidad del Aire con Machine Learning
# 
# **Autor:** [Emiliano Rodriguez Villegas](https://github.com/EmilianoRdzV)
# **Fecha:** 06 de Junio de 2025
# **Versión:** 1.0
# 
# ---
# 
# ## 1. Introducción y Motivación
# 
# Este notebook presenta un análisis completo de la calidad del aire en seis metrópolis globales. El objetivo es aplicar un flujo de trabajo de ciencia de datos, desde la limpieza de datos hasta la construcción de un modelo de Machine Learning para clasificar la calidad del aire.
# 
# La elección de este conjunto de datos se basa en su **volumen**, la **diversidad geográfica** de las ciudades, la **riqueza de sus características** y su **componente temporal**, que lo hacen ideal para el análisis y la modelización.
# 
# ---
# 
# ### Índice General del Notebook
# 
# 1.  [**Metodología y Preparación**](#fase-1)
#     * [1.1. Diagrama de Flujo del Proyecto](#1-1)
#     * [1.2. Carga de Datos desde URL](#1-2)
#     * [1.3. Preprocesamiento y Limpieza](#1-3)
#     * [1.4. Análisis Estadístico Descriptivo](#1-4)
# 2.  [**Modelado de Clasificación**](#fase-2)
#     * [2.1. Preparación de Datos para el Modelo](#2-1)
#     * [2.2. Entrenamiento del Modelo (Perceptrón / Adaline)](#2-2)
#     * [2.3. Visualización de Resultados](#2-3)
# 3.  [**Discusión y Trabajo Futuro**](#fase-3)
#     * [3.1. Análisis del Modelo y Propuestas de Mejora](#3-1)
#     * [3.2. Comparación con la Literatura](#3-2)
#     * [3.3. Trabajo Futuro](#3-3)

# ## <a id="fase-1"></a>1. Metodología y Preparación
# 
# ### <a id="1-1"></a>1.1. Diagrama de Flujo del Proyecto
# 
# A continuación, se presenta un diagrama de flujo que resume visualmente la metodología y los pasos a seguir en este proyecto, desde la adquisición de los datos hasta la obtención de conclusiones.

# ![Diagrama de Flujo del Proyecto](../Data/FTrabajo.png)

# ### <a id="1-2"></a>1.2. Carga de Datos desde URL
# 
# 
# Se procederá a cargar el conjunto de datos en un DataFrame de `pandas`. URL de la DB: https://www.kaggle.com/datasets/youssefelebiary/air-quality-2024/data
# 
# 
# *** Se intento leer los datos directamente desde una URL pero por el volumen de datos no se realizo, no todas las DBs lo permiten, asi que se opto por descargar la DB y trabajarla desde el alojamiennto local en ../Data/Air_Quality.csv

# ### Carga de Datos




rutaDatos = '../Data/Air_Quality.csv'
dataFrame = pd.read_csv(rutaDatos)

#Info bascia para observar el paronama de los datos
print (dataFrame.head())


# ### <a id="1-3"></a>1.3. Preprocesamiento y Limpieza
# 
# 
# En esta etapa, abordaremos los problemas comunes en los datos crudos para asegurar su calidad. El proceso se dividirá en:
# 1.  Identificar la cantidad de valores nulos (NaN).
# 2.  Imputar (rellenar) los valores nulos con una estrategia adecuada.
# 3.  Verificar y eliminar filas duplicadas.
# 
# **Justificación del Procedimiento:** No se eliminarán las filas con valores NaN directamente, ya que esto podría resultar en una pérdida significativa de datos secuenciales, lo cual es vital para el análisis de series temporales. En su lugar, se optará por la **interpolación lineal**, un método que estima un valor faltante basándose en los valores numéricos que lo rodean, asumiendo una progresión constante entre ellos. Esta es una técnica robusta para datos de sensores como los de calidad del aire.



# 1: Revisión de Valores Nulos (NaN)
# Contamos cuántos valores nulos hay en cada columna para entender la magnitud del problema.
print("* Conteo de Valores Nulos ANTES del preprocesamiento")
print(dataFrame.isnull().sum())

# 2: Imputación de Valores por Interpolación Lineal
# Rellenamos los valores NaN usando el método de interpolación lineal.
dataFrame.interpolate(method='linear', inplace=True)
dataFrame.fillna(method='bfill', inplace=True)

print("\n* Conteo de Valores Nulos DESPUÉS de la imputación")
print(dataFrame.isnull().sum())




# 3: Verificación y Eliminación de Duplicados
print(f"\nNúmero de filas duplicadas encontradas: {dataFrame.duplicated().sum()}")

# Fila duplicada, se elimina.
dataFrame.drop_duplicates(inplace=True)

print("\n Preprocesamiento y limpieza de datos comletado.")


# ### Corregir el formato de la columna de fecha 
# Convertiremos la columna de fecha al tipo `datetime`, despues se convertira en el index del Data Frame 



# Columna de texto a un objeto de fecha y hora (datetime)
dataFrame['Date'] = pd.to_datetime(dataFrame['Date'])

# Establece la columna 'Date' como el nuevo índice del DataFrame
dataFrame.set_index('Date', inplace=True)

# Verificamos que el cambio se haya realizado correctamente
print("--- Info después de procesar la fecha ---")
dataFrame.info()

print("\n--- DataFrame con índice de fecha ---")
display(dataFrame.head())


# ### Guardamos el Data Frame ya procesado



# Ruta para el nuevo archivo CSV limpio
rutaArchivoLimpio = '../Data/AirQualityCleaned.csv'

# index=True asegura que la columna de fecha se guarde en el archivo
dataFrame.to_csv(rutaArchivoLimpio, index=True)

print(f"Los datos limpios se han guardado en: '{rutaArchivoLimpio}'")


# ### <a id="1-4"></a>1.4. Análisis Estadístico Descriptivo
# 
# Una vez que los datos están limpios, es útil obtener un resumen numérico de alto nivel de todas las características. Utilizaremos la función `.describe()` 



# Estadísticas descriptiva
# Media, desviación estándar, valores mínimos y máximos, y los percentiles.
estadisticas = dataFrame.describe()

display(estadisticas)


# ## <a id="fase-2"></a>2. Fase 2: Análisis Exploratorio de Datos (EDA)
# 
# Con los datos ya limpios, podemos comenzar a explorarlos para entender sus características principales. En esta fase, buscaremos patrones y relaciones a través de resúmenes estadísticos y, fundamentalmente, visualizaciones gráficas. `matplotlib` `seaborn`
# 
# ---
# 
# ### <a id="2-1"></a>2.1. Análisis Descriptivo General
# 
# Aunque ya vimos una tabla de estadísticas en la fase anterior, las visualizaciones nos permiten entender la **distribución** de los datos de una manera más intuitiva. Crearemos histogramas para las columnas más importantes (AQI y los principales contaminantes) para observar cómo se reparten sus valores: si son simétricos, si tienen sesgos, etc.




# Configurar el estilo de los gráficos para que se vean mejor
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Columnas más relevantes para visualizar
columnasVisualizar = ['AQI', 'PM2.5', 'PM10', 'NO2', 'CO', 'O3']

# Histograma para cada columna seleccionada
dataFrame[columnasVisualizar].hist()

plt.tight_layout()
plt.show()

