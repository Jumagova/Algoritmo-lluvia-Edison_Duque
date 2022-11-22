# Algoritmo lluvia

Script de Python, para identificar grabaciones acústicas con lluvia. Se cuenta con la posibilidad de utilizar la versión propuesta por DUQUE,2019 Y la versión de EDISON,2022.

## Como utilizar este algoritmo desarroyado en Python?

Desde la terminal de anaconda, ejecutamos la siguiente instrucción de codigo, para crear una variable de entorno de python llamada ***algoritmolluvia***


```python

conda create --name algoritmolluvia python=3.10.6

```

Lluego activamos, la variable de entorno ***algoritmolluvia*** mediante la siguente instrucción

```python
conda activate algoritmolluvia
```
Despues de ejecutar el comando debemos verificar que el ambiente se activo, para esto debemos fijarnos en la terminal donde al principio de la linea debe indicar el ambiente sobre el cual estamos trabajando, como se ve a continuación.

```python
(algoritmolluvia) $ >>
```
A continuación, instalamos en la variable de entorno ***algoritmolluvia*** las librerías necesarias para correr el algoritmo en Pytnon sin problemas, mediante las siguientes instrucciones

```python
(algoritmolluvia) $ >> pip install soundfile 
(algoritmolluvia) $ >> conda install pandas
(algoritmolluvia) $ >> conda install numpy 
(algoritmolluvia) $ >> pip install scipy
(algoritmolluvia) $ >> conda install -c conda-forge tqdm
(algoritmolluvia) $ >> pip install openpyxl
```

Finalmente, nos ubicamos en la ruta donde se encuentra el archivo ***cambiar_ext.py***, el cual contiene el algoritmo que deseamos utilizar

```python
cd ubicacion_del_archivo_cambiar_ext.py
```

Una vez, estamos en la ruta donde se encuentra el archivo ***cambiar_ext.py***, ejecutamos el siguiente comando. Donde el ***path***, es la ruta de la carpeta raiz que contiene los archivos que queremos modificar.

```python
(algoritmoExt) $ >> python cambiar_ext.py -p 'path'
```

