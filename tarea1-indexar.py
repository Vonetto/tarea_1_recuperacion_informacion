# tarea1-indexar.py:
# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 11 de agosto de 2024
# Alumno: [Juan Vicente Onetto Romero]

import sys
import os
import util as util
import numpy as np
from skimage import io, color
from skimage.transform import resize
from tqdm import tqdm
import cv2  # Asegúrate de tener OpenCV instalado

def calcular_histograma_color(imagen, bins=(8, 8, 8)):
    """
    Calcula el histograma de color de una imagen.
    
    Parámetros:
    - imagen: Imagen en escala de grises o color.
    - bins: Número de bins para cada canal de color.
    
    Retorna:
    - Histograma de color normalizado como un vector 1D.
    """
    if len(imagen.shape) == 2:  # Escala de grises
        hist = np.histogram(imagen, bins=bins[0], range=(0, 1))[0]
    else:
        # Calcular histograma para cada canal y concatenar
        hist = []
        for channel in range(imagen.shape[2]):
            hist_channel = np.histogram(imagen[:, :, channel], bins=bins[channel], range=(0, 1))[0]
            hist.append(hist_channel)
        hist = np.concatenate(hist)
    
    # Normalizar el histograma
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def calcular_descriptor(imagen_path, tamaño=(20, 20)):
    """
    Calcula un descriptor de intensidades para una imagen dada.
    
    Parámetros:
    - imagen_path: Ruta de la imagen.
    - tamaño: Tupla indicando el tamaño al que se redimensionará la imagen.
    
    Retorna:
    - Descriptor de intensidades como un vector 1D.
    """
    try:
        imagen = io.imread(imagen_path)
        if len(imagen.shape) == 3:
            imagen_gray = color.rgb2gray(imagen)
        else:
            imagen_gray = imagen
        imagen_resized = resize(imagen_gray, tamaño, anti_aliasing=True)
        descriptor_intensidad = imagen_resized.flatten()
        
        # Normalizar el descriptor
        descriptor_intensidad = descriptor_intensidad / (np.linalg.norm(descriptor_intensidad) + 1e-7)
        
        return descriptor_intensidad
    except Exception as e:
        print(f"Error procesando {imagen_path}: {e}")
        return None

def tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R, tamaño=(20, 20)):
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
    
    # Listas para almacenar descriptores y nombres de imágenes
    descriptores_R = []
    nombres_R = []
    
    # Procesar cada imagen
    for imagen in tqdm(lista_imagenes, desc="Indexando imágenes R"):
        ruta_imagen = os.path.join(dir_input_imagenes_R, imagen)
        descriptor = calcular_descriptor(ruta_imagen, tamaño)
        if descriptor is not None:
            descriptores_R.append(descriptor)
            nombres_R.append(imagen)
        else:
            print(f"Descriptor no calculado para {imagen}")
        
        # Aplicar Flip Horizontal
        try:
            imagen_cv = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
            if imagen_cv is not None:
                imagen_flip = cv2.flip(imagen_cv, 1)  # Flip horizontal
                imagen_flip_resized = cv2.resize(imagen_flip, tamaño, interpolation=cv2.INTER_AREA)
                descriptor_flip = imagen_flip_resized.flatten()
                descriptor_flip = descriptor_flip / (np.linalg.norm(descriptor_flip) + 1e-7)
                descriptores_R.append(descriptor_flip)
                nombres_R.append(imagen)  # Asociar flip con el nombre original
            else:
                print(f"No se pudo leer la imagen para flip: {imagen}")
        except Exception as e:
            print(f"Error al aplicar flip a {imagen}: {e}")
    
    # Convertir a numpy array
    descriptores_R = np.array(descriptores_R, dtype=np.float32)
    
    # Guardar los descriptores y nombres en un único archivo .pkl
    archivo_salida = os.path.join(dir_output_descriptores_R, "descriptores_R.pkl")
    data_to_save = {
        'descriptores': descriptores_R,
        'nombres': nombres_R
    }
    util.guardar_objeto(data_to_save, dir_output_descriptores_R, "descriptores_R.pkl")
    print(f"Descriptores guardados en {archivo_salida}")

# Inicio de la tarea
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Uso: python {sys.argv[0]} [dir_input_imagenes_R] [dir_output_descriptores_R]")
        sys.exit(1)

    dir_input_imagenes_R = sys.argv[1]
    dir_output_descriptores_R = sys.argv[2]

    tarea1_indexar(dir_input_imagenes_R, dir_output_descriptores_R)
