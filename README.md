# DMA-eigenfaces

Estas son las librerías necesarias para correr todo:

* Análisis de datos: `numpy`, `pandas`, `matplotlib`

* Procesamiento de imágenes: `opencv-python`, `pillow-heif`

Este es el orden en el que hay que correr las cosas:

`cd helper_scripts/`

`python convert_heic.py`

`python face_extraction.py`

`python mirror_images.py`

`cd ..`

`python eigenfaces.py`

`python neural_network.py`
