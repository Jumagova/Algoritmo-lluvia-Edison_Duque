# Cambiar nombre y extensión de archivos .WAV

Script de Python, para modificar el nombre y la extensión de archivos. Identifica todos los archivos dentro de la carpeta raiz y cambia la extensión de los archivos a minuscula ejemplo ('.WAV' a '.wav'). Luego, modifica el nombre a cada archivo, agregando un prefijo, correspondiente a la carpeta al que pertence.

## Como utilizar este algoritmo desarroyado en Python?

Desde la terminal de anaconda, ejecutamos la siguiente instrucción de codigo, para crear una variable de entorno de python llamada ***algoritmoExt***


```python

conda create --name algoritmoExt python=3.10.6

```

Lluego activamos, la variable de entorno ***algoritmoExt*** mediante la siguente instrucción

```python
conda activate algoritmoExt
```
Despues de ejecutar el comando debemos verificar que el ambiente se activo, para esto debemos fijarnos en la terminal donde al principio de la linea debe indicar el ambiente sobre el cual estamos trabajando, como se ve a continuación.

```python
(algoritmoExt) $ >>
```



Finalmente, nos ubicamos en la ruta donde se encuentra el archivo ***cambiar_ext.py***, el cual contiene el algoritmo que deseamos utilizar

```python
cd ubicacion_del_archivo_cambiar_ext.py
```

Una vez, estamos en la ruta donde se encuentra el archivo ***cambiar_ext.py***, ejecutamos el siguiente comando. Donde el ***path***, es la ruta de la carpeta raiz que contiene los archivos que queremos modificar.

```python
(algoritmoExt) $ >> python cambiar_ext.py -p 'path'
```

