# tarea1-buscar.py:
# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 11 de agosto de 2024
# Alumno: [Tu Nombre]

import sys
import os
import util as util
import numpy as np
from skimage import io, color
from skimage.transform import resize
from tqdm import tqdm
import cv2  # Asegúrate de tener OpenCV instalado
from scipy.spatial.distance import cdist

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

def calcular_descriptor(imagen_path, tamaño=(5, 5)):
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

def tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados):
    if not os.path.isdir(dir_input_imagenes_Q):
        print(f"ERROR: no existe directorio {dir_input_imagenes_Q}")
        sys.exit(1)
    elif not os.path.isdir(dir_input_descriptores_R):
        print(f"ERROR: no existe directorio {dir_input_descriptores_R}")
        sys.exit(1)
    elif os.path.exists(file_output_resultados):
        print(f"ERROR: ya existe archivo {file_output_resultados}")
        sys.exit(1)
    
    # Cargar descriptores de R
    archivo_descriptores_R = os.path.join(dir_input_descriptores_R, "descriptores_R.pkl")
    if not os.path.isfile(archivo_descriptores_R):
        print(f"ERROR: no se encontró {archivo_descriptores_R}")
        sys.exit(1)
    
    print("Cargando descriptores de R...")
    data_R = util.leer_objeto(dir_input_descriptores_R, "descriptores_R.pkl")
    descriptores_R = data_R['descriptores']
    nombres_R = data_R['nombres']
    print(f"Total de descriptores en R: {len(descriptores_R)}")
    
    # Listar imágenes Q
    lista_imagenes_Q = util.listar_archivos_en_carpeta(dir_input_imagenes_Q)
    print(f"Total de imágenes Q a procesar: {len(lista_imagenes_Q)}")
    
    resultados = []
    
    # Convertir descriptores_R a numpy array si no lo está
    if not isinstance(descriptores_R, np.ndarray):
        descriptores_R = np.array(descriptores_R, dtype=np.float32)
    
    # Procesar cada imagen Q
    for imagen_Q in tqdm(lista_imagenes_Q, desc="Buscando imágenes Q"):
        ruta_imagen_Q = os.path.join(dir_input_imagenes_Q, imagen_Q)
        descriptor_Q = calcular_descriptor(ruta_imagen_Q)
        if descriptor_Q is not None:
            # Aplicar Flip Horizontal a Q
            try:
                imagen_cv_Q = cv2.imread(ruta_imagen_Q, cv2.IMREAD_GRAYSCALE)
                if imagen_cv_Q is not None:
                    imagen_flip_Q = cv2.flip(imagen_cv_Q, 1)  # Flip horizontal
                    imagen_flip_resized_Q = cv2.resize(imagen_flip_Q, (5, 5), interpolation=cv2.INTER_AREA)
                    descriptor_flip_Q = imagen_flip_resized_Q.flatten()
                    descriptor_flip_Q = descriptor_flip_Q / (np.linalg.norm(descriptor_flip_Q) + 1e-7)
                else:
                    print(f"No se pudo leer la imagen para flip: {imagen_Q}")
                    descriptor_flip_Q = None
            except Exception as e:
                print(f"Error al aplicar flip a {imagen_Q}: {e}")
                descriptor_flip_Q = None
            
            # Si se pudo calcular el descriptor flip, considerar ambos
            if descriptor_flip_Q is not None:
                # Combinar descriptors Q original y flip
                descriptores_Q = np.vstack([descriptor_Q, descriptor_flip_Q])
            else:
                descriptores_Q = np.vstack([descriptor_Q])
            
            # Calcular distancias entre Q y R usando Euclidean
            # descriptores_Q es (num_descriptores_Q, descriptor_length)
            # descriptores_R es (num_descriptores_R, descriptor_length)
            # Usar cdist para eficiencia
            distancia = cdist(descriptores_Q, descriptores_R, metric='euclidean')
            
            # Encontrar la distancia mínima y el índice correspondiente
            min_distancias = distancia.min(axis=1)
            min_indices = distancia.argmin(axis=1)
            
            # Seleccionar la mínima distancia entre Q y R
            min_distancia = min_distancias.min()
            min_index = min_indices[np.argmin(min_distancias)]
            imagen_R_mas_cercana = nombres_R[min_index]
            
            resultados.append([imagen_Q, imagen_R_mas_cercana, min_distancia])
        else:
            print(f"Descriptor no calculado para {imagen_Q}")
    
    # Escribir resultados en archivo de salida
    print(f"Escribiendo resultados en {file_output_resultados}...")
    util.escribir_lista_de_columnas_en_archivo(resultados, file_output_resultados)
    print("Búsqueda completada.")

# Inicio de la tarea
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Uso: python {sys.argv[0]} [dir_input_imagenes_Q] [dir_input_descriptores_R] [file_output_resultados]")
        sys.exit(1)

    dir_input_imagenes_Q = sys.argv[1]
    dir_input_descriptores_R = sys.argv[2]
    file_output_resultados = sys.argv[3]

    tarea1_buscar(dir_input_imagenes_Q, dir_input_descriptores_R, file_output_resultados)
