# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:04:39 2020

@author: usuario
"""
# #############################################################################
# Cargar scripts propios

from aux_functions import *

# #############################################################################
# Cargar librerías

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    
    path = "C:/Users/DAVID/Desktop/TESIS/Proyectos/NCH/Implementacion/Prueba1/CURVAS"
    
    AE_results = cargar_resultados_txt(path+"/AE_results.txt")
    IF_results = cargar_resultados_txt(path+"/IF_results.txt")
    #LOF_results = cargar_resultados_txt(path+"/LOF_results.txt")
    OCSVM_results = cargar_resultados_txt(path+"/OCSVM_results.txt")
    RC_results = cargar_resultados_txt(path+"/RC_results.txt")
    #SVDD_results = cargar_resultados_txt(path+"/SVDD_results.txt")
    NCH_results = cargar_resultados_txt(path+"/NCH_results.txt")

    AE = parsear_y_calcular_metricas(AE_results)
    IF = parsear_y_calcular_metricas(IF_results)
    #LOF = parsear_y_calcular_metricas(LOF_results)
    OCSVM = parsear_y_calcular_metricas(OCSVM_results)
    RC = parsear_y_calcular_metricas(RC_results)
    #SVDD = parsear_y_calcular_metricas(SVDD_results)
    NCH = parsear_y_calcular_metricas(NCH_results)
    
    
    indice = 3
    print(obtener_mejor_metodo(AE, indice))
    print("")
    print(obtener_mejor_metodo(IF, indice))
    print("")
    #print(obtener_mejor_metodo(LOF, indice))
    print(obtener_mejor_metodo(OCSVM, indice))
    print("")
    print(obtener_mejor_metodo(RC, indice))
    print("")
    #print(obtener_mejor_metodo(SVDD, indice))
    print(obtener_mejor_metodo(NCH, indice))
    
    
    

    
    
    
    
    
    
    
    
    
    
    