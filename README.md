# Algoritmo lluvia

Script de Python para identificar grabaciones acústicas con lluvia. Se cuenta con la posibilidad de utilizar la versión propuesta por DUQUE, 2019 y la versión de EDISON, 2022.

## Cómo utilizar este algoritmo desarrollado en Python con Docker?

### Paso 1: Compilar la imagen del contenedor

Para utilizar el algoritmo en un contenedor Docker, primero debes compilar la imagen del contenedor. Ejecuta el siguiente comando en la terminal para compilar la imagen:

```bash
./build
```

Esto creará una imagen Docker con el nombre `lluvia_filter` utilizando el archivo Dockerfile proporcionado.

### Paso 2: Ejecutar el contenedor

Una vez que la imagen del contenedor ha sido compilada, puedes ejecutar el contenedor Docker para utilizar el algoritmo. Ejecuta el siguiente comando en la terminal:

```bash
./run
```

El comando anterior ejecutará el contenedor y automáticamente ejecutará el algoritmo de detección de lluvia con los parámetros predeterminados dentro del contenedor.

### Cambiar el comando de ejecución

Si deseas cambiar el comando de ejecución del script de Python dentro del contenedor, debes modificar el archivo Dockerfile. A continuación se muestra cómo puedes hacerlo:

1. Abre el archivo `Dockerfile` en un editor de texto.

2. Busca la línea que comienza con `CMD python lluvia_plateromodEdison.py` y modifica los argumentos según tus necesidades.

Donde:
- `E_D` es una variable booleana para utilizar la versión del algoritmo lluvia EDISON,2022 (`True`) o DUQUE,2019 (`False`).
- `path` es la ruta de la carpeta raíz que contiene las grabaciones acústicas que deseas identificar con lluvia.
- `patharch` es la ruta donde deseas guardar el archivo Excel con los resultados.
- `name` es el nombre del archivo Excel con extensión `.xlsx`.
- `seg` es el tiempo de duración de las grabaciones en segundos con el que deseas trabajar.

Ejemplo de comando utilizando la versión del algoritmo lluvia EDISON,2022 y trabajando solo con grabaciones de 60 segundos:

```
CMD  python lluvia_plateromodEdison.py -Edison_Duque True -p 'grabaciones' -pr 'resultados' -name 'ResultGrab.xlsx' -seg 60
```

Por ejemplo, si deseas utilizar la versión del algoritmo lluvia DUQUE,2019 y trabajar con grabaciones de 120 segundos, el comando se vería así:

``
CMD python lluvia_plateromodEdison.py -Edison_Duque False -p 'grabaciones' -pr 'resultados' -name 'ResultGrab.xlsx' -seg 120
```

Asegúrate de ajustar los valores de los argumentos según tus requisitos.

3. Guarda los cambios en el archivo `Dockerfile`.

4. Vuelve a compilar la imagen del contenedor utilizando el comando `./build`.

Una vez que hayas realizado estos cambios y vuelto a compilar la imagen del contenedor, al ejecutar el comando `./run`, el nuevo comando de ejecución del script de Python se utilizará dentro del contenedor.
