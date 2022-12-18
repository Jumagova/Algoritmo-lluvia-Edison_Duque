# Algoritmo lluvia

Script de Python, para identificar grabaciones acústicas con lluvia. Se cuenta con la posibilidad de utilizar la versión propuesta por DUQUE,2019 y la versión de EDISON,2022.

## Como utilizar este algoritmo desarrollado en Python?

Desde la terminal de anaconda, ejecutamos la siguiente instrucción de código, para crear una variable de entorno de Python llamada ***algoritmolluvia***


```Python

conda create --name algoritmolluvia python=3.10.6

```

Luego activamos, la variable de entorno ***algoritmolluvia*** mediante la siguiente instrucción

```python
conda activate algoritmolluvia
```
Después de ejecutar el comando debemos verificar que el ambiente se activó, para esto debemos fijarnos en la terminal donde al principio de la línea debe indicar el ambiente sobre el cual estamos trabajando, como se ve a continuación.

```Python
(algoritmolluvia) $ >>
```
A continuación, instalamos en la variable de entorno ***algoritmolluvia*** las librerías necesarias para correr el algoritmo en Python sin problemas, mediante las siguientes instrucciones

```Python
(algoritmolluvia) $ >> pip install soundfile 
(algoritmolluvia) $ >> conda install pandas
(algoritmolluvia) $ >> conda install numpy 
(algoritmolluvia) $ >> pip install scipy
(algoritmolluvia) $ >> conda install -c conda-forge tqdm
(algoritmolluvia) $ >> pip install openpyxl
```

Finalmente, nos ubicamos en la ruta donde se encuentra el archivo ***lluvia_plateromodEdison.py***, el cual contiene el algoritmo que deseamos utilizar

```Python
cd ubicacion_del_archivo_lluvia_plateromodEdison.py
```

Una vez, estamos en la ruta donde se encuentra el archivo ***lluvia_plateromodEdison.py***, ejecutamos el siguiente comando. Donde ***E_D*** es una variable booleana para utilizar la versión del algoritmo lluvia EDISON,2022 (True) y DUQUE,2019 (False). El ***path*** es la ruta de la carpeta raíz que contiene las carpetas con las grabaciones acústicas que queremos identificar con lluvia. El ***patharch*** es la ruta donde vamos a guardar el archivo Excel con los resultados. El ***name*** es el nombre del archivo Excel con extensión .xlsx y ***seg***  es el tiempo de duración de las grabaciones en segundos con la que se decea trabajar.

```Python
(algoritmolluvia) $ >> python lluvia_plateromodEdison.py -Edison_Duque 'E_D' -p 'path' -pr 'patharch' -name 'name'  -seg 'seg'
```
Ejemplo del comando anterior, utilizando la versión del algoritmo lluvia EDISON,2022 y trabajando solo con grabaciones de 60 segundos.

```Python
(algoritmolluvia) $ >> python lluvia_plateromodEdison.py -Edison_Duque 'True' -p 'C:\Users\grabaciones_prueba' -pr 'C:\Users\Algoritmo_result' -name 'ResultGrab.xlsx' -seg '60'
```

