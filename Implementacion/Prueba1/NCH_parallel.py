# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:57:12 2020

@author: DAVID
"""
from calcular_NCH_simple import *
from calcular_NCH_simple_con_graficas import *
from aux_functions import *
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import multiprocessing as mp
import time
import ast


def NCH_train (dataset, n_projections, l, extend, contraer_SCH, threads):
    # Generamos las proyecciones bidimensionales
    projections = generate_Projections(n_projections, dataset.shape[1])
    # Proyectamos los datos en estos espacios 2D
    dataset_projected = project_Dataset(dataset, projections)
    
    # Calculamos el NCH y SNCH en cada proyección
    l_vertices = []
    l_aristas = []
    l_vertices_expandidos = []
    l_orden_vertices = []
    l_factor_expansion = []       
    
    arguments_iterable = []
    for i in range (0, n_projections):
        arguments_iterable.append((dataset_projected[i], l, extend, contraer_SCH))
        
    tic = time.perf_counter()    
    process_pool = mp.Pool(threads)
    result = process_pool.starmap(calcular_NCH_simple, arguments_iterable)
    process_pool.close()
    process_pool.join()
    
    toc = time.perf_counter()
    for i in range (0, n_projections):
        arguments_iterable.append((dataset_projected[i], l, extend, contraer_SCH))
        l_vertices.append(result[i][0])
        l_aristas.append(result[i][1])
        l_vertices_expandidos.append(result[i][2])
        l_orden_vertices.append(result[i][3])
        l_factor_expansion.append(result[i][4]) 

    
    print("-------------")
    #print("l_factor_expansion", l_factor_expansion)
    # Chekear si todos los factores de expansion son el mismo para emplear el NCH o el SNCH
    if ((len(np.unique(l_factor_expansion))) != 1):
        l_vertices_expandidos = False
        print("Los factores de expansion de las distintas proyecciones son diferentes por lo que no se va a emplear el cierre escalado.")
    elif (l_factor_expansion == [0]):
        l_vertices_expandidos = False
        print("Solo se ha seleccionado una única proyección y NO empleará el cierre expandido ya que es complejo.")
    elif (((np.unique(l_factor_expansion)) == 0) and (contraer_SCH == True)):
        l_vertices_expandidos = False
        print("Se han seleccionado varias proyecciones pero NO emplearán sus cierres expandidos ya que son complejos.")    
    print("-------------")  
    return projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices


def NCH_classify (dataset, model, threads):        
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices = model
    
     

    # Proyectamos los datos a clasificar
    tic = time.perf_counter() 
    dataset_projected = project_Dataset(dataset, projections)
    
    tic = time.perf_counter()
    result = check_if_points_are_inside_polygons_p(dataset_projected, model, threads)
    toc = time.perf_counter()
    """
    tic = time.perf_counter()
    result2 = check_if_points_are_inside_polygons(dataset_projected, model)
    toc = time.perf_counter()
    print("Tiempo 2: %0.4f segundos" % (toc - tic)) 
    """
    result = combinar_clasificaciones(result) 
    
    return result