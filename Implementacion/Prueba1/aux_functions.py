# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:25:52 2020

@author: DAVID
"""

import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import math
import multiprocessing as mp
import matplotlib.path as mpltPath

# ##################################################################
# Entrenamiento estandarizador

def NormalizeData_Train(dataframe_proccessed):
    # Entrenamos un normalizador de media cero y desviacion tipica 1 con el dataframe de entrada
    #print("- - -")
    #print("Starting normalizer train...")
    scaler = preprocessing.StandardScaler().fit(dataframe_proccessed) 
    #print("Trained.")
    return scaler

# ##################################################################
# Aplicación de un estandarizador
    
def NormalizeDataframe(dataframe_proccessed, model):
    #print("- - -")
    #print("Starting normalization...")
    columns = dataframe_proccessed.columns
    # Normalizamos el dataframe de entrada mediante el normalizador recibido como argumento
    data = model.transform(dataframe_proccessed.astype(float))
    data = pd.DataFrame(data, columns=columns)
    #print("Data normalized.")
    return data

def NormalizeData(data, model):
    #print("- - -")
    #print("Starting normalization...")
    # Normalizamos el dataframe de entrada mediante el normalizador recibido como argumento
    data = model.transform(data)
    #print("Data normalized.")
    return data

# ##################################################################
# Inversión de la normalización

def inverse_transform(dataframe_proccessed, model):
    print("- - -")
    print("Starting inverse transformation...")
    columns = dataframe_proccessed.columns
    data = model.inverse_transform(dataframe_proccessed.astype(float))
    data = pd.DataFrame(data, columns=columns)
    print("Data transformed...")
    return data.to_numpy()

def change_target_value_GH(df):
    if df == 'g':
        return 0
    elif df == 'h':
        return 1
    
def change_target_value_01(df):
    if df == 1:
        return 0
    elif df == -1:
        return 1
 
def change_target_value_MNIST(df):
    if df == 1 or df == 7:
        return 1
    else:
        return 0
    
def array_to_sequence_of_vertices (data):
    from ground.base import get_context
    
    context = get_context()    
    Point = context.point_cls
    Contour = context.contour_cls

    aux_list = []
    for i in range (0, data.shape[0]):
        #aux_list.append((data[i, 0],data[i, 1]))
        aux_list.append(Point(data[i, 0], data[i, 1]))
    
    aux_list = Contour(aux_list)
    
    return aux_list

def array_to_sequence_of_vertices2 (data):
    # Función auxiliar para transformar una matriz de numpy de vértices en una lista con el formato [(X1,Y1), (X2,Y2), ... , (Xn,Yn)]

    aux_list = []
    for i in range (0, data.shape[0]):
        aux_list.append((data[i, 0],data[i, 1]))

    return aux_list


def generate_Projections (n_projections, n_dim):
    # Función que genera n matrices de proyección bidimensionales
    import numpy as np
    projections = np.random.randn(n_dim, 2, n_projections)
    return projections    


def project_Dataset (dataset, projections):
    # Función que proyecta un conjunto de datos a partir de matrices de proyección
    
    import numpy as np
    n_projections = projections.shape[2]
    dataset_projected = []
    for j in range(0, n_projections):
        one_projection = np.matmul(dataset, projections[:, :, j])
        dataset_projected.append(one_projection)
    return dataset_projected

def check_if_points_are_inside_polygons (dataset, model):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices = model
    
    l_results = []
    
    if (dataset[0].ndim == 1):
        num_datos = 1
    else:
        num_datos = dataset[0].shape[0]

    for i in range (0, len(l_vertices)):
        aux = []
        print("Proy:", i)
        if (l_vertices_expandidos != False): # Si los cierres SI se expandieron durante el entrenamiento, utilizamos el SNCH para clasificar
            # Construimos el polígono a partir de los vértices del SNCH
            polygon = Polygon(array_to_sequence_of_vertices2(l_vertices_expandidos[i]))
            
        elif (l_vertices_expandidos == False): # Si los cierres NO se expandieron durante el entrenamiento, utilizamos el NCH para clasificar
            # Construimos el polígono a partir de los vértices del NCH
            polygon = Polygon(array_to_sequence_of_vertices2(l_vertices[i][l_orden_vertices[i]]))
            
        for j in range (0, num_datos): # Clasificamos cada uno de los puntos
            #1 print("Dato:", j)    
            if (num_datos == 1):
                point = Point(dataset[i])
            else:
                point = Point(dataset[i][j])
                
            aux.append(polygon.contains(point)) # Lista de comprobaciones para una proyección
            #1 print("Dentro: ", polygon.contains(point))
        
        l_results.append(aux) # Lista de listas de proyecciones
    
    return l_results

def check_if_points_are_inside_polygons_p (dataset, model, process_pool):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices = model
    if (dataset[0].ndim == 1):
        num_datos = 1
    else:
        num_datos = dataset[0].shape[0]

    arguments_iterable = []
    for i in range (0, projections[0].shape[1]):
        if (l_vertices_expandidos != False):
            parameter = l_vertices_expandidos[i]
        else:
            parameter = l_vertices[i][l_orden_vertices[i]]
        arguments_iterable.append((l_vertices_expandidos, l_vertices[i], parameter, num_datos, dataset[i]))
        
    #process_pool = mp.Pool(threads)
    result = list(process_pool.imap(check_one_projection, arguments_iterable))
    #process_pool.close()
    #process_pool.terminate()
    #process_pool.join()
    
    #process_pool.close()
    #process_pool.restart()
    
    return result

def check_one_projection(args):   
    l_vertices_ex, vertices, l_vertices_x, n_datos, dataset = args
    aux = []
    
    # Construimos el polígono a partir de los vértices del NCH
    polygon = Polygon(array_to_sequence_of_vertices2(l_vertices_x))
        
    for j in range (0, n_datos): # Clasificamos cada uno de los puntos
        #1 print("Dato:", j)    
        if (n_datos == 1):
            point = Point(dataset)
        else:
            point = Point(dataset[j])
            
        aux.append(polygon.contains(point)) # Lista de comprobaciones para una proyección
        #1 print("Dentro: ", polygon.contains(point))
    
    return aux # Lista de listas de proyecciones

def check_if_points_are_inside_polygons_matplotlib (dataset, model, process_pool):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    
    projections, l_vertices, _, l_vertices_expandidos, l_orden_vertices, _ = model
    if (dataset[0].ndim == 1):
        num_datos = 1
    else:
        num_datos = dataset[0].shape[0]

    arguments_iterable = []
    for i in range (0, projections[0].shape[1]):
        if (l_vertices_expandidos != False):
            parameter = l_vertices_expandidos[i]
        else:
            parameter = l_vertices[i][l_orden_vertices[i]]

        arguments_iterable.append((l_vertices_expandidos, l_vertices[i], parameter, num_datos, dataset[i]))
        
    result = list(process_pool.imap(check_one_projection_matplotlib, arguments_iterable))

    return result

def check_if_points_are_inside_polygons_matplotlib_sin_paralelizar (dataset, model):
    # Función que determina si uno o varios datos pasados como matriz de numpy se encuentran dentro de un modelo NCH entrenado
    aux = []
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices, _ = model
    
    if (dataset[0].ndim == 1):
        num_datos = 1
    else:
        num_datos = dataset[0].shape[0]

    for i in range (0, len(l_vertices)):
        print("Proy:", i)
        if (l_vertices_expandidos != False): # Si los cierres SI se expandieron durante el entrenamiento, utilizamos el SNCH para clasificar
            # Construimos el polígono a partir de los vértices del SNCH
            polygon = mpltPath.Path(vertices = l_vertices_expandidos[i]) 

        else: # Si los cierres NO se expandieron durante el entrenamiento, utilizamos el NCH para clasificar
            # Construimos el polígono a partir de los vértices del NCH
            polygon = mpltPath.Path(vertices = l_vertices[i][l_orden_vertices[i]]) 
        
        clasificacion = list(polygon.contains_points(dataset[i]))
        aux.append(clasificacion)
    
    return aux

def check_one_projection_matplotlib(args):   
    l_vertices_ex, vertices, l_vertices_x, n_datos, dataset = args
    aux = []
    # Construimos el polígono a partir de los vértices del NCH
    
    polygon = mpltPath.Path(vertices = l_vertices_x) 

    aux = polygon.contains_points(dataset)
        
    return aux # Lista de listas de proyecciones

def combinar_clasificaciones(result):
    # Funcion que recibe una lista de listas y combinar los resultados -> si un dato es clasificado en alguna proyección
    # como anómalo, el resultado será anómalo
    n_proyections = len(result)
    n_datos = len(result[0])
    
    combination = np.full(n_datos, -1)
    
    for i in range (0, n_datos):
        aux = []
        for j in range (0, n_proyections):
            aux.append(result[j][i])
        if (sum(aux) < n_proyections) :
            combination[i] = 1
        else:
            combination[i] = 0

    return combination

def calcular_metricas (Y_test, result, titulo):
    cm = confusion_matrix(Y_test, result).ravel()
    if cm.shape[0] > 1:
        TN, FP, FN, TP = confusion_matrix(Y_test, result).ravel()
    else:
        if Y_test.iloc[0] == 0:
            TN, FP, FN, TP = Y_test.shape[0], 0, 0, 0
        elif Y_test.iloc[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, Y_test.shape[0]
        
    print("")
    print(titulo)
    print("-TN, FP, FN, TP: ", TN, FP, FN, TP)
    #print("-Sensibilidad TP/(TP+FN): ", TP/(TP+FN))
    #print("-Especificidad TN/(TN+FP): ", TN/(TN+FP))
    #print("-Precisión (TP+TN)/(TP+TN+FP+FN): ", (TP+TN)/(TP+TN+FP+FN))
    #print("-Similitud: ", 1-(math.sqrt((1-(TP+TN)/(TP+TN+FP+FN))**2+(1-TP/(TP+FN))**2)/math.sqrt(2)))
    print("")
    return [TN, FP, FN, TP]
    
def cargar_resultados_txt (path):
    import ast
    lines = []
    with open(path, "r") as reader:
        lines.append(reader.readline())
    return ast.literal_eval(lines[0])

def weird_division(n, d):
    return n / d if d else 0


def parsear_y_calcular_metricas (list_results):
    desired_output = []
    for result in list_results:
        TN, FP, FN, TP, info = result 
        sensibilidad = TP/(TP+FN)
        especificidad = TN/(TN+FP)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        similitud = 1-(math.sqrt((1-(TP+TN)/(TP+TN+FP+FN))**2+(1-TP/(TP+FN))**2)/math.sqrt(2))
        precision = weird_division(TP, TP+FP)
        F1 = weird_division((2*precision*sensibilidad), (precision+sensibilidad))
        
        desired_output.append([sensibilidad, especificidad, accuracy, similitud, F1, info])
    return desired_output

def obtener_mejor_metodo (list_results, index_metric):
    
    if index_metric != -1:
        list_results_target_metric = []
        for result in list_results:
            list_results_target_metric.append(result[index_metric])
        
    else:
        list_results_target_metric = []
        for result in list_results:
            list_results_target_metric.append((result[0] + result[1])/2)

    max_value = max(list_results_target_metric)
    max_index = list_results_target_metric.index(max_value)
    return list_results[max_index]


