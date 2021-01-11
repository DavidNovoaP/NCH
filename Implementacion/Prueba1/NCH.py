# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 10:57:12 2020

@author: DAVID
"""
from calcular_NCH_simple import *
#from calcular_NCH_simple_con_graficas import *
from aux_functions import *
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



def NCH_train (dataset, n_projections, l, extend, contraer_SCH):
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
    
    
    print("-------------")
    print("Projection: ", 0) 
    vertices_aux, aristas_aux, vertices_expandidos, orden_vertices_aux, factor_expansion_aux =  calcular_NCH_simple((dataset_projected[0], l, extend, contraer_SCH))
    l_vertices.append(vertices_aux)
    l_aristas.append(aristas_aux)
    l_vertices_expandidos.append(vertices_expandidos)
    l_orden_vertices.append(orden_vertices_aux)
    l_factor_expansion.append(factor_expansion_aux)
        
    for i in range (1, n_projections):
        print("-------------")
        print("Projection: ", i) 
        vertices_aux, aristas_aux, vertices_expandidos, orden_vertices_aux, factor_expansion_aux =  calcular_NCH_simple((dataset_projected[i], l, extend, contraer_SCH))
        l_vertices.append(vertices_aux)
        l_aristas.append(aristas_aux)
        l_vertices_expandidos.append(vertices_expandidos)
        l_orden_vertices.append(orden_vertices_aux)
        l_factor_expansion.append(factor_expansion_aux)
        
        
    print("-------------")
    print("l_factor_expansion", np.unique(l_factor_expansion))
    # Chekear si todos los factores de expansion son el mismo para emplear el NCH o el SNCH
    if (((len(np.unique(l_factor_expansion))) > 1) and (contraer_SCH == True)):
        l_vertices_expandidos = False
        print("Los factores de expansion de las distintas proyecciones son diferentes por lo que no se va a emplear el cierre escalado.")
    elif (l_factor_expansion == [0]):
        l_vertices_expandidos = False
        print("Solo se ha seleccionado una única proyección y NO empleará el cierre expandido ya que es complejo.")
    elif ((len(np.unique(l_factor_expansion)) == 1) and ((np.unique(l_factor_expansion)) == 0) and (contraer_SCH == True)):
        l_vertices_expandidos = False
        print("Se han seleccionado varias proyecciones pero NO emplearán sus cierres expandidos ya que son complejos.")    
    print("-------------")  
    return projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices


def NCH_classify (dataset, model):        
    projections, l_vertices, l_aristas, l_vertices_expandidos, l_orden_vertices = model
    
    # Proyectamos los datos a clasificar
    print("proyectar datos")
    tic = time.perf_counter() 
    # Proyectamos los datos a clasificar
    dataset_projected = project_Dataset(dataset, projections)
    toc = time.perf_counter() 
    print("tiempo: ", toc-tic)
    
    print("check dentro/fuera")
    tic = time.perf_counter() 
    result = check_if_points_are_inside_polygons(dataset_projected, model)
    toc = time.perf_counter() 
    print("tiempo: ", toc-tic)
    
    print("combinar")
    tic = time.perf_counter()
    result = combinar_clasificaciones(result) 
    toc = time.perf_counter() 
    print("tiempo: ", toc-tic)
    print("-------------")

    
    return result