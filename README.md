# DMA-eigenfaces

Estas son las librerías necesarias para correr todo:

* Análisis de datos: `numpy`, `pandas`, `matplotlib`

* Procesamiento de imágenes: `opencv-python`, `pillow-heif`

## Evaluacion de caras nuevas

Solo se necesitan las librerias `json`, `numpy` y `pandas` para probar nuevas imagenes

Para poder evaluar nuevas caras, 

* Se deberan copiar las imagenes a la carpeta `4.test_your_files/` y correr el script de python `Z_test_new_files.py`

* En la consola devolvera el nombre del archivo evaluado y la prediccion de la persona a la que corresponde la imagen

* Las imagenes preprocesadas se guardan en la carpeta `4.test_your_files/preprocessed_test_files/` para verificar la correcta deteccion de la cara en la imagen

## Entrenamiento de la red

Este es el orden en el que hay que correr las cosas y entrenar a la red:

`cd helper_scripts/`

`python convert_heic.py` --> Convierte imagenes en formato HEIC a jpg

`python face_extraction.py` --> Extrae las caras de las imagenes

`cd ..`

`python eigenfaces.py` --> Genera las componentes princpales y exporta la matrix para que entrenar la red

`python Z_neural_network.py` --> Entrena la red sobre las componentes principales exportadas. Al concluir el entrenamiento se puede decidir guardar los parametros de la red
