import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['KAGGLE_CONFIG_DIR'] = '/content'

# Descargar y descomprimir CelebA de forma silenciosa para que no colapse la RAM
!kaggle datasets download -d jessicali9530/celeba-dataset --unzip -q

# Configurar el aumento de datos sintéticos (como pide tu tarea)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Construir la red convolucional base
modelo_base = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(40, activation='sigmoid') # 40 neuronas de salida para los 40 atributos de CelebA
])

modelo_base.summary()
