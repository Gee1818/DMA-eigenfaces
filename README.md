# DMA-eigenfaces

Estas son las librerías necesarias para correr todo:

* Análisis de datos: `numpy`, `pandas`, `matplotlib`

* Procesamiento de imágenes: `opencv-python`, `pillow-heif`

## Evaluacion de caras nuevas

Solo se necesitan las librerias `numpy` y `pandas` para probar nuevas imagenes

Para poder evaluar nuevas caras, 

* Se deberan copiar las imagenes a la carpeta `4.test_your_files/` y correr el script de python `Z_test_new_files.py`

* En la consola devolvera el nombre del archivo evaluado y la prediccion del nombre

* Las imagenes preprocesadas se guardan en la carpeta `4.test_your_files/preprocessed_test_files/` para verificar la correcta deteccion de la cara en la imagen

## Entrenamiento de la red

Este es el orden en el que hay que correr las cosas y entrenar a la red:

`cd helper_scripts/`

`python convert_heic.py`

`python face_extraction.py`

`python mirror_images.py`

`python brightness.py`

`cd ..`

`python eigenfaces.py`

`python neural_network.py`
