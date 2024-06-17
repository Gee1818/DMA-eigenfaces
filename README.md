# DMA-eigenfaces

Estas son las librerías necesarias para correr todo:

* Análisis de datos: `numpy`, `pandas`, `matplotlib`

* Procesamiento de imágenes: `opencv-python`, `pillow-heif`

## Evaluación de caras nuevas

Sólo se necesitan las librerías `json`, `numpy` y `pandas` para probar nuevas imágenes

Para poder evaluar nuevas caras, 

* Se deberán copiar las imágenes a la carpeta `4.test_your_files/` y correr el script de Python `05_test_new_files.py`

* La consola devolverá el nombre del archivo evaluado y la predicción de la persona a la que corresponde la imagen

* Las imágenes preprocesadas se guardan en la carpeta `4.test_your_files/preprocessed_test_files/` para verificar la correcta detección de la cara en la imagen

## Entrenamiento de la red

Este es el orden en el que hay que correr las cosas y entrenar a la red:

`python 01_convert_heic.py` (15 segundos) --> Convierte imágenes en formato HEIC a jpg

`python 02_face_extraction.py` (6 minutos) --> Extrae las caras de las imágenes

`python 03_eigenfaces.py` (2 segundos) --> Genera las componentes princpales y exporta la matrix para que entrenar la red

`python 04_neural_network.py` (5 segundos) --> Entrena la red sobre las componentes principales exportadas. Al concluir el entrenamiento se puede decidir guardar los parámetros de la red
