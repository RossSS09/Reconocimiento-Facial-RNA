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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Construir un nuevo modelo con las capas convolucionales del modelo base (quitando las densas)
modelo_transfer = Sequential()

# Extraemos las primeras 6 capas del modelo_base (las Conv2D y MaxPooling2D)
for capa in modelo_base.layers[:6]:
    modelo_transfer.add(capa)

# Congelar los pesos de la parte pre-entrenada para no arruinarlos con el gradiente inicial
for capa in modelo_transfer.layers:
    capa.trainable = False

# Aumentar un clasificador apropiado para distinguir sólo el rostro
modelo_transfer.add(Flatten())
modelo_transfer.add(Dense(128, activation='relu')) # Capa densa oculta 
modelo_transfer.add(Dense(1, activation='sigmoid')) # Una sola neurona de salida (clasificación binaria)

modelo_transfer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Verificar la arquitectura final y comprobar que los parámetros pre-entrenados no son entrenables (Non-trainable params)
modelo_transfer.summary()


import os
import shutil

# Crear estructura de carpetas para las dos categorías
base_dir = '/content/dataset_rostros'
mi_rostro_dir = os.path.join(base_dir, 'mi_rostro')
otros_rostros_dir = os.path.join(base_dir, 'otros_rostros')

os.makedirs(mi_rostro_dir, exist_ok=True)
os.makedirs(otros_rostros_dir, exist_ok=True)

# Mover 100 imágenes aleatorias de CelebA a la carpeta de 'otros_rostros' para equilibrar los datos
celeba_source_dir = '/content/img_align_celeba/img_align_celeba'

if os.path.exists(celeba_source_dir):
    imagenes_celeba = os.listdir(celeba_source_dir)[:100] # Tomamos 100 fotos para el ejemplo
    for img_name in imagenes_celeba:
        src = os.path.join(celeba_source_dir, img_name)
        dst = os.path.join(otros_rostros_dir, img_name)
        shutil.copy(src, dst)
    print("Carpeta 'otros_rostros' lista con imágenes de CelebA.")
else:
    print("Revisa la ruta de CelebA, no se encontró la carpeta extraída.")

print("Carpetas creadas con éxito. Ahora debes subir tus fotos a la carpeta 'mi_rostro'.")


# Paso 5: Entrenar el modelo
# Conectar el generador de datos con la carpeta que contiene las dos categorías
train_generator = datagen.flow_from_directory(
    '/content/dataset_rostros',
    target_size=(224, 224), # Mismo tamaño de entrada de la red convolucional
    batch_size=16,
    class_mode='binary' # Clasificación binaria: es tu rostro o no lo es
)

print("Iniciando el entrenamiento...")
# Entrenar el modelo con las capas pre-entrenadas congeladas
historial = modelo_transfer.fit(
    train_generator,
    epochs=10, # Número de pasadas completas sobre los datos
    steps_per_epoch=len(train_generator)
)
print("¡Entrenamiento finalizado!")
