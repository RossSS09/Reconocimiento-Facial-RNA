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
