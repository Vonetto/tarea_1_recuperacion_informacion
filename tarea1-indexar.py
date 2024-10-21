# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 11 de agosto de 2024
# Alumno: [Juan Vicente Onetto Romero]

import sys
import os
import util as util
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
import pickle
from tqdm import tqdm



def calcular_hog_descriptor(imagen_path, tamaño=(128, 128)):
    """
    Calcula el descriptor HOG para una imagen dada.
    
    Parámetros:
    - imagen_path: Ruta de la imagen.
    - tamaño: Tupla indicando el tamaño al que se redimensionará la imagen.
    
    Retorna:
    - descriptor HOG como un vector 1D.
    """
    try:
        imagen = io.imread(imagen_path)
        if len(imagen.shape) == 3:
            imagen = color.rgb2gray(imagen)
        imagen = resize(imagen, tamaño, anti_aliasing=True)
        # Asegurarse de que visualize=False para que solo se devuelva el descriptor
        descriptor = hog(imagen, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), feature_vector=True, visualize=False)
        return descriptor
    except Exception as e:
        print(f"Error procesando {imagen_path}: {e}")
        return None


def tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R):
    if not os.path.isdir(dir_input_imagenes_R):
        print(f"ERROR: no existe directorio {dir_input_imagenes_R}")
        sys.exit(1)
    elif os.path.exists(dir_output_descriptores_R):
        print(f"ERROR: ya existe directorio {dir_output_descriptores_R}")
        sys.exit(1)
    
    # Crear la carpeta de salida
    os.makedirs(dir_output_descriptores_R, exist_ok=True)
    
    # Listar todas las imágenes .jpg en R
    lista_imagenes = util.listar_archivos_en_carpeta(dir_input_imagenes_R)
    print(f"Total de imágenes a procesar: {len(lista_imagenes)}")
    
    # Diccionario para almacenar descriptores
    descriptores_R = {}
    
    # Procesar cada imagen
    for imagen in tqdm(lista_imagenes, desc="Indexando imágenes R"):
        ruta_imagen = os.path.join(dir_input_imagenes_R, imagen)
        descriptor = calcular_hog_descriptor(ruta_imagen)
        if descriptor is not None:
            descriptores_R[imagen] = descriptor
        else:
            print(f"Descriptor no calculado para {imagen}")
    
    # Guardar todos los descriptores en un único archivo .pkl
    archivo_salida = os.path.join(dir_output_descriptores_R, "descriptores_R.pkl")
    util.guardar_objeto(descriptores_R, dir_output_descriptores_R, "descriptores_R.pkl")
    print(f"Descriptores guardados en {archivo_salida}")

# Inicio de la tarea
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Uso: python {sys.argv[0]} [dir_input_imagenes_R] [dir_output_descriptores_R]")
        sys.exit(1)

    dir_input_imagenes_R = sys.argv[1]
    dir_output_descriptores_R = sys.argv[2]

    tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R)