import soundfile as sf
import numpy as np
import pandas as pd
from scipy import signal, stats
import os
import tqdm
from multiprocessing import Pool
import time
from datetime import timedelta
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def meanspec(audio, Fs= 1, wn="hann", ovlp=0, wl=512, nfft = None, norm=True):

    '''
    Calcula el espectro medio haciendo el promedio en el eje de las frecuencias del espectrograma.

    :param audio: señal monoaural temporal (numpy array)
    :param Fs: frecuencia de muestreo en Hz, valor por defecto 1 (int)
    :param wn: tipo de ventana, valor por defecto "hann" (str)
    :param ovlp: puntos de solapamiento entre ventanas, valor por defecto 0 (int)
    :param wl: tamaño de la ventana, valor por defecto 512 (int)
    :param nfft: número de puntos de la transformada de Fourier, valor por defecto, None, es decir el mismo de wl (int)
    :param norm: booleano que indica si se normaliza o no el espectro, valor por defecto, True.
    :return: array con el rango de frecuencias y la señal con el espectro medio (numpy array)
    '''

    f, t, Zxx = signal.stft(audio, fs = Fs, window=wn, noverlap=ovlp, nperseg=wl, nfft=nfft)


    mspec = np.mean(np.abs(Zxx), axis=1)

    if norm == True:
        mspec = mspec/max(mspec)

    return f, mspec

def calculo_PSD_promedio(df_ll, pbar=None):
        
    '''
    Calcula la densidad espectral de potencia (PSD) media y utiliza la funcion meanspec para calcula el espectro medio.

    :param df_ll: ruta de la grabacion que se esta analizando y carpeta a la que pertenece
    :return: ruta de la grabacion, directorio en con rangos de frecuencia vs PDS encontrado, valor PSD media, indicador para saber si el archivo esta corrupto, nombre de la carpeta al que pertenece la grabacion
    '''

    canal=0
    fmin = 200
    fmax = 11250
    tipo_ventana = "hann"
    sobreposicion = 0
    tamano_ventana = 1024
    nfft = tamano_ventana
    banda_lluvia = (600, 1200)
    
    ruta_archivo = df_ll.path_FI
    grupo = df_ll.field_number_PR
    duracion_escog1 = df_ll.duracion_escog
    duracion_escog1 = int(duracion_escog1)

    try:
        x, Fs = sf.read(ruta_archivo) 

        if len(x.shape) == 1:
            audio = x
        else:
            audio = x[:, canal]

        puntos_minuto = Fs * 60
        npuntos = len(audio)
        Duracon_Gra = int(npuntos/Fs)

        banda = []

        for seg in range(0, npuntos, puntos_minuto):
            f, p = signal.welch(audio[seg:puntos_minuto+seg], Fs, nperseg=512, window=tipo_ventana,
                                nfft=512, noverlap=sobreposicion)
            banda.append(p[np.logical_and(f >= banda_lluvia[0], f <= banda_lluvia[1])])

        banda = np.concatenate(banda)

        if duracion_escog1==Duracon_Gra:
            PSD_medio = np.mean(banda)

            if tamano_ventana > Fs // 2:
                raise NotImplementedError("Ventana demasiado grande")
            else:
                nfft = tamano_ventana
                
            f, mspec = meanspec(audio, Fs, tipo_ventana, sobreposicion, tamano_ventana, nfft)
            
            cond = np.logical_and(f > fmin, f < fmax)
            feats = list(mspec[cond])
            freqs = list(f[cond])
            titulos = [f"mPSD_{int(freqs[i])}" for i in range(len(freqs))]
                    
            zip_iterator = zip(titulos, feats)
            
            if pbar is not None:
                pbar.update(1)

            medianaPDS = np.median(mspec[cond])
            if medianaPDS > 0.9:
                #print(f"El archivo {ruta_archivo} tiene alto PSD y tienen una media {mediaPDS} y una mediana {medianaPDS}.")
                return ruta_archivo, {}, 0, 'NO_ALTO_PSD', grupo,  Duracon_Gra
            else:
                return ruta_archivo, dict(zip_iterator), PSD_medio, 'NO', grupo, Duracon_Gra

        else:
            #print(f"El archivo {ruta_archivo} tiene una duración diferente.")
            return ruta_archivo, {}, 0, 'NO_DIF', grupo,  Duracon_Gra


    except:
        if pbar is not None:
            pbar.update(1)
        print(f"El archivo {ruta_archivo} esta corrupto.")
        return ruta_archivo, {}, 0, 'YES', grupo, 'NA'

def calculo_PSD_and_Espectro_promedio(df_ll, pbar=None):

    '''
    Calcula la densidad espectral de potencia (PSD) media y utiliza la funcion meanspec para calcula el espectro medio.

    :param df_ll: ruta de la grabacion que se esta analizando y carpeta a la que pertenece
    :return: ruta de la grabacion, directorio en con rangos de frecuencia vs PDS encontrado, valor PSD media, indicador para saber si el archivo esta corrupto, nombre de la carpeta al que pertenece la grabacion y espectrograma promedio de la grabacion
    '''

    
    canal=0
    canal1=1
    fmin = 200
    fmax = 11250
    tipo_ventana = "hann"
    sobreposicion = 0
    tamano_ventana = 1024
    nfft = tamano_ventana
    banda_lluvia = (600, 1200)
    
    ruta_archivo = df_ll.path_FI
    grupo = df_ll.field_number_PR
    duracion_escog1 = df_ll.duracion_escog
    duracion_escog1 = int(duracion_escog1)
    
    try:
        x, Fs = sf.read(ruta_archivo) 
        if len(x.shape) == 1:
            audio = x
            audio1 = x
        else:
            audio = x[:, canal]
            audio1 = x[:, canal1]

        puntos_minuto = Fs * 60
        npuntos = len(audio)
        Duracon_Gra = int(npuntos/Fs)

        banda = []

        for seg in range(0, npuntos, puntos_minuto):
            f, p = signal.welch(audio[seg:puntos_minuto+seg], Fs, nperseg=512, window=tipo_ventana,
                                nfft=512, noverlap=sobreposicion)
            banda.append(p[np.logical_and(f >= banda_lluvia[0], f <= banda_lluvia[1])])

        banda = np.concatenate(banda)

        if duracion_escog1==Duracon_Gra:

            PSD_medio = np.mean(banda)

            if tamano_ventana > Fs // 2:
                raise NotImplementedError("Ventana demasiado grande")
            else:
                nfft = tamano_ventana
                
            f, mspec = meanspec(audio, Fs, tipo_ventana, sobreposicion, tamano_ventana, nfft)
            
            cond = np.logical_and(f > fmin, f < fmax)
            feats = list(mspec[cond])
            freqs = list(f[cond])
            titulos = [f"mPSD_{int(freqs[i])}" for i in range(len(freqs))]
                    
            zip_iterator = zip(titulos, feats)

            # se obtiene el espectrograma de la grabacion
            f, t, s = signal.spectrogram(audio1, Fs, window=tipo_ventana,nfft=512, 
                                            mode="magnitude"
                                            )
            meanspectro = (s.mean(axis=1))                         # se guarda el espectrograma promedio de la grabacion
            
            if pbar is not None:
                pbar.update(1)

            medianaPDS = np.median(mspec[cond])
            if medianaPDS > 0.9:
                #print(f"El archivo {ruta_archivo} tiene alto PSD y tienen una media {mediaPDS} y una mediana {medianaPDS}.")
                info_grab_aux = np.zeros(shape=(1,257))
                return ruta_archivo, {}, 0, 'NO_ALTO_PSD', grupo, info_grab_aux[0], Duracon_Gra
            else:
                return ruta_archivo, dict(zip_iterator), PSD_medio, 'NO', grupo, meanspectro, Duracon_Gra
        
        else:
            info_grab_aux = np.zeros(shape=(1,257))
            #print(f"El archivo {ruta_archivo} tiene una duración diferente.")
            return ruta_archivo, {}, 0, 'NO_DIF', grupo, info_grab_aux[0], Duracon_Gra

    
    except:
        if pbar is not None:
            pbar.update(1)
        info_grab_aux = np.zeros(shape=(1,257))
        print(f"El archivo {ruta_archivo} esta corrupto.")
        return ruta_archivo, {}, 0, 'YES', grupo, info_grab_aux[0], 'NA'
    
    

def _apply_df(args):
    df, func = args
    res = df.apply(func, axis=1)
    return res

def regla_decision(x, umbral):
    if x != 0:
        if x < umbral:
            return "NO"
        elif x >= umbral:
            return "YES"
        else:
            raise NotImplementedError
    else:
        return "PSD medio 0"

def algoritmo_lluvia_imp(df_ind):

    '''

    Esta función filtra las grabaciones con altos niveles de ruido.

    Además se genera un umbral automático para el reconocimiento de las grabaciones más ruidosas.

    :param df_ind: dataframe que contiene la informacion de las grabaciones corruptas y el valor PSD media
    :return: dataframe con el nombre de cada carpeta, grabacion, valores de PSD media en cada rango de frecuencia e indicador para saber si la grabacion contiene lluvia "YES" O "NO"  
    '''

    df_lluvia = df_ind.loc[df_ind.damaged_FI == 'NO', :].copy()
    df_no_lluvia = df_ind.loc[df_ind.damaged_FI == 'YES', :].copy()
    df_time_dif = df_ind.loc[df_ind.damaged_FI == 'NO_DIF', :].copy()
    df_alto_psd = df_ind.loc[df_ind.damaged_FI == 'NO_ALTO_PSD', :].copy()

    PSD_medio = np.array(df_lluvia.PSD_medio.values).astype(np.float64)
    PSD_medio_sin_ceros = PSD_medio[PSD_medio > 0]
    umbral = (np.mean(PSD_medio_sin_ceros) + stats.mstats.gmean(PSD_medio_sin_ceros)) / 2
    
    df_lluvia['rain_FI'] = df_lluvia.PSD_medio.apply(regla_decision, umbral=umbral)
    df_lluvia = df_lluvia.drop(['PSD_medio'], axis=1)
    
    df_no_lluvia['rain_FI'] = 'Archivo corrupto'
    df_no_lluvia = df_no_lluvia.drop(['PSD_medio'], axis=1)

    df_time_dif['rain_FI'] = 'Tiempo diferente'
    df_time_dif = df_time_dif.drop(['PSD_medio'], axis=1)

    df_alto_psd['rain_FI'] = 'ALTO PSD'
    df_alto_psd = df_alto_psd.drop(['PSD_medio'], axis=1)

    df_indices_lluvia =  pd.concat([df_lluvia, df_no_lluvia, df_time_dif, df_alto_psd])

    assert df_indices_lluvia.shape[0] == df_ind.shape[0]

    return df_indices_lluvia

def algoritmo_lluvia_imp_intensidad(df_ind, arraymeanspect_ind):

    '''

    Esta función filtra las grabaciones con altos niveles de ruido.

    Además se generan umbral automático para el reconocimiento de las grabaciones más ruidosas.

    :param df_ind: dataframe que contiene la informacion de las grabaciones corruptas y el valor PSD media
    :return: dataframe con el nombre de cada carpeta, grabacion, valores de PSD media en cada rango de frecuencia e indicador para saber si la grabacion contiene lluvia "YES" O "NO"  
    '''

    df_lluvia = df_ind.loc[df_ind.damaged_FI == 'NO', :].copy()
    df_no_lluvia = df_ind.loc[df_ind.damaged_FI == 'YES', :].copy()
    df_time_dif = df_ind.loc[df_ind.damaged_FI == 'NO_DIF', :].copy()
    df_alto_psd = df_ind.loc[df_ind.damaged_FI == 'NO_ALTO_PSD', :].copy()

    arraymeanspect_lluvia = arraymeanspect_ind[df_ind.damaged_FI == 'NO']
    arraymeanspect_no_lluvia = arraymeanspect_ind[df_ind.damaged_FI == 'YES']

    PSD_medio = np.array(df_lluvia.PSD_medio.values).astype(np.float64)
    PSD_medio_sin_ceros = PSD_medio[PSD_medio > 0]
    umbral = (np.mean(PSD_medio_sin_ceros) + stats.mstats.gmean(PSD_medio_sin_ceros)) / 2
    
    df_lluvia['rain_FI_PSD'] = df_lluvia.PSD_medio.apply(regla_decision, umbral=umbral)
    df_lluvia = df_lluvia.drop(['PSD_medio'], axis=1)
    
    df_no_lluvia['rain_FI_PSD'] = 'Archivo corrupto'
    df_no_lluvia = df_no_lluvia.drop(['PSD_medio'], axis=1)

    df_time_dif['rain_FI_PSD'] = 'Tiempo diferente'
    df_time_dif = df_time_dif.drop(['PSD_medio'], axis=1)

    df_alto_psd['rain_FI_PSD'] = 'ALTO PSD'
    df_alto_psd = df_alto_psd.drop(['PSD_medio'], axis=1)

    arraymeanspect_lluvia_umbral = arraymeanspect_lluvia[df_lluvia.rain_FI_PSD == 'YES']

    nivel_actual = np.median(arraymeanspect_lluvia_umbral[:,43:65])

    nivle_Normalizacion = 3.814165176863326e-05                                # mediana ideal de todo el grupo de grabaciones
    diferencia_escala = nivel_actual/nivle_Normalizacion            
    arraymeanspect_lluvia = arraymeanspect_lluvia/diferencia_escala            # se lleva todo el grupo de espectrogramas promedio de todas las grabaciones a la escala ideal

    array_grab_lluvia = []
    # se procesa cada grabacion para identificar cuales contiene lluvia
    for i in range(len(arraymeanspect_lluvia[:,0])):
        max_0_1500hz = max(arraymeanspect_lluvia[i,0:17])
        des_0_24000hz =  np.std(arraymeanspect_lluvia[i])
        max_9000_24000hz = max(arraymeanspect_lluvia[i,96:258])
        des_9000_24000hz = np.std(arraymeanspect_lluvia[i,96:258])
        min_1781_2343hz = min(arraymeanspect_lluvia[i,19:26])

        if max_0_1500hz > 0.0001:
            array_grab_lluvia.append("YES")    # grabacion con lluvia fuerte
        elif  max_9000_24000hz >= 0.0002:
            array_grab_lluvia.append("NO")    # grabacion sin lluvia 
        elif des_9000_24000hz > 0.00001:
            array_grab_lluvia.append("NO")    # grabacion sin lluvia
        elif ((des_0_24000hz <= 2.3e-06) | (des_0_24000hz >= 4e-5)):
            array_grab_lluvia.append("NO")    # grabacion sin lluvia
        elif min_1781_2343hz < 0.00001:
            array_grab_lluvia.append("NO")    # grabacion sin lluvia
        else:
            array_grab_lluvia.append("YES")    # grabacion con lluvia leve

    df_lluvia['rain_FI'] = array_grab_lluvia
    df_no_lluvia['rain_FI'] = 'Archivo corrupto'
    df_time_dif['rain_FI'] = 'Tiempo diferente'
    df_alto_psd['rain_FI'] = 'ALTO PSD'

    df_indices_lluvia =  pd.concat([df_lluvia, df_no_lluvia, df_time_dif, df_alto_psd])

    assert df_indices_lluvia.shape[0] == df_ind.shape[0]

    return df_indices_lluvia

if __name__ == '__main__':

    #Edison_Duque = True # toma valor True para utilizar metodo de EDISON,2022 y False para utilizar metodo de DUQUE,2019

    argparser = argparse.ArgumentParser(description='Corre algoritmo lluvia')
    argparser.add_argument('-Edison_Duque', '--E_D',required=True, help='Variable booleana para utilizar algoritmo lluvia, True (Edison) y False (Duque)')
    argparser.add_argument('-p', '--path',required=True, help='Ruta donde se encuentran los archivos')
    argparser.add_argument('-pr', '--patharch',required=True, help='Ruta donde se guardan los resultados')
    argparser.add_argument('-name', '--name',required=True, help='Nombre del archivo de resultados con extension .xlsx')
    argparser.add_argument('-seg', '--seg',required=True, help='Tiempo de duración de la grabación en segundos')

    args = argparser.parse_args()

    Edison_Duque = args.E_D
    ruta_datos = args.path
    folder_rain  = args.patharch
    name_file = args.name
    Duration=args.seg

    #name_file = 'Resultado_lluvia_platero_Tenero_Audible1.xlsx'

    formatos = ['wav', 'WAV']
    n_cores = 14
    exclude_these_sites = []

    dict_df = {"field_number_PR" : [],
           "name_FI" : [],
           "path_FI" : []}

    print(f"Inventorying Files...")
    start_time = time.time()
    for (root, dirs, file) in os.walk(ruta_datos):
        for f in file:
            if any([f".{formato}" in f for formato in formatos]):
                if not(f.startswith(".")):
                    dict_df["field_number_PR"].append(os.path.basename(root))
                    dict_df["name_FI"].append(f)
                    dict_df["path_FI"].append(os.path.join(root, f))
    df = pd.DataFrame(dict_df)
    df[['prefix', 'date', 'hour', 'format']] = df.name_FI.str.extract(r'(?P<prefix>\w+)_(?P<date>\d{8})_(?P<hour>\d{6}).(?P<format>[0-9a-zA-Z]+)')
    df[['duracion_escog']] = Duration

    print(f"{len(df)} files found")

    df =  df.loc[df.field_number_PR.apply(lambda x : x not in exclude_these_sites), :]
    
    print(f"{len(df)} files found after exclude {','.join(exclude_these_sites)} sites")

    print(f"Calculating Indices...")
    workers = min(len(df), n_cores)
    
    if(8 <= len(df) // workers):
        fact_split = 8
    else:
        fact_split = 1

    df_split = np.array_split(df, fact_split * workers)

    if(Edison_Duque=='True'):
        with Pool(processes=workers) as pool:
            result = list(tqdm.tqdm(pool.imap(_apply_df, [(d, calculo_PSD_and_Espectro_promedio) for d in df_split]), total=len(df_split)))
        
        
        x = pd.concat(result)

        print(f"Running Rain Algorithm...")
        x = np.array(list(zip(*x))).T

        df_ind = pd.DataFrame(list(x[:, 1]))
        df_ind['path_FI'] = x[:, 0]
        df_ind['PSD_medio'] = x[:, 2]
        df_ind['damaged_FI'] = x[:, 3]
        df_ind['grupo'] = x[:, 4]
        df_ind['Duracion(seg)'] = x[:, 6]

        
        meanspect_aux = []                                                         
        for id_espect in range(len(x[:, 5])):                                     
            meanspect_aux.append(x[id_espect, 5])                                 
        arraymeanspect_aux = np.array(meanspect_aux)                               
        arraymeanspect_aux = arraymeanspect_aux.reshape(len(x[:, 5]), 257)         
        
        
        df_lluvias = [] 
        for i in tqdm.tqdm(df_ind['grupo'].unique()):
            df_tmp = df_ind[df_ind.grupo == i]
            arraymeanspect_tmp = arraymeanspect_aux[df_ind.grupo == i]                      
            df_lluvias.append(algoritmo_lluvia_imp_intensidad(df_tmp,arraymeanspect_tmp))   

        
        df_indices_lluvia = pd.concat(df_lluvias)

        assert len(df) == len(df_indices_lluvia)

        df_y = df.merge(df_indices_lluvia, how='left', on='path_FI')
        df_y = df_y.drop(['path_FI', 'date', 'prefix', 'hour', 'format', 'damaged_FI', 'grupo', 'rain_FI_PSD', 'duracion_escog'], axis=1) 
        cols = list((df_y.columns))
        cols2 =  cols[:-2] + cols[-1:] + ['Duracion(seg)']
        df_y = df_y[cols2]

        path_file = os.path.join(folder_rain, name_file)

        print(f"Saving in {path_file} ...")
        df_y.to_excel(path_file, index=False)
        print(f"Results saved in {path_file}")
        print(f"Execution Time {str(timedelta(seconds=time.time() - start_time))}")

    else:
        with Pool(processes=workers) as pool:
            result = list(tqdm.tqdm(pool.imap(_apply_df, [(d, calculo_PSD_promedio) for d in df_split]), total=len(df_split)))
             
        x = pd.concat(result)

        print(f"Running Rain Algorithm...")
        x = np.array(list(zip(*x))).T

        df_ind = pd.DataFrame(list(x[:, 1]))
        df_ind['path_FI'] = x[:, 0]
        df_ind['PSD_medio'] = x[:, 2]
        df_ind['damaged_FI'] = x[:, 3]
        df_ind['grupo'] = x[:, 4]
        df_ind['Duracion(seg)'] = x[:, 5]
        
        
        df_lluvias = [] 
        for i in tqdm.tqdm(df_ind['grupo'].unique()):
            df_tmp = df_ind[df_ind.grupo == i]
            df_lluvias.append(algoritmo_lluvia_imp(df_tmp))
        
        df_indices_lluvia = pd.concat(df_lluvias)

        assert len(df) == len(df_indices_lluvia)

        df_y = df.merge(df_indices_lluvia, how='left', on='path_FI')
        df_y = df_y.drop(['path_FI', 'date', 'prefix', 'hour', 'format', 'damaged_FI', 'grupo', 'duracion_escog'], axis=1)
        cols = list((df_y.columns))
        cols2 =  cols[:-2] + cols[-1:] + ['Duracion(seg)']
        df_y = df_y[cols2]
        
        path_file = os.path.join(folder_rain, name_file)

        print(f"Saving in {path_file} ...")
        df_y.to_excel(path_file, index=False)
        print(f"Results saved in {path_file}")
        print(f"Execution Time {str(timedelta(seconds=time.time() - start_time))}")