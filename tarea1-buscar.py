# CC5213 - TAREA 1 - RECUPERACIÓN DE INFORMACIÓN MULTIMEDIA
# 11 de agosto de 2024
# Alumno: [Tu Nombre]

import sys
import os
import util as util
import numpy as np
from skimage import io, color
from skimage.transform import resize
from skimage.feature import hog
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
        descriptor = hog(imagen, pixels_per_cell=(16, 16),
                         cells_per_block=(2, 2), feature_vector=True)
        return descriptor
    except Exception as e:
        print(f"Error procesando {imagen_path}: {e}")
        return None


def calcular_distancia_euclidiana(descriptor1, descriptor2):
    """
    Calcula la distancia euclidiana entre dos descriptores.
    
    Parámetros:
    - descriptor1: Vector de características de la imagen 1.
    - descriptor2: Vector de características de la imagen 2.
    
    Retorna:
    - La distancia euclidiana entre descriptor1 y descriptor2.
    """
    return np.linalg.norm(descriptor1 - descriptor2)

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
    descriptores_R = util.leer_objeto(dir_input_descriptores_R, "descriptores_R.pkl")
    print(f"Total de descriptores en R: {len(descriptores_R)}")
    
    # Listar imágenes Q
    lista_imagenes_Q = util.listar_archivos_en_carpeta(dir_input_imagenes_Q)
    print(f"Total de imágenes Q a procesar: {len(lista_imagenes_Q)}")
    
    resultados = []
    
    # Procesar cada imagen Q
    for imagen_Q in tqdm(lista_imagenes_Q, desc="Buscando imágenes Q"):
        ruta_imagen_Q = os.path.join(dir_input_imagenes_Q, imagen_Q)
        descriptor_Q = calcular_hog_descriptor(ruta_imagen_Q)
        if descriptor_Q is not None:
            # Encontrar la imagen R más cercana
            imagen_R_mas_cercana = None
            distancia_minima = float('inf')
            
            for imagen_R, descriptor_R in descriptores_R.items():
                distancia = calcular_distancia_euclidiana(descriptor_Q, descriptor_R)
                if distancia < distancia_minima:
                    distancia_minima = distancia
                    imagen_R_mas_cercana = imagen_R
            
            if imagen_R_mas_cercana is not None:
                resultados.append([imagen_Q, imagen_R_mas_cercana, distancia_minima])
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
