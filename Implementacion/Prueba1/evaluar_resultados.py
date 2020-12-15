# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 10:04:39 2020

@author: usuario
"""
# #############################################################################
# Cargar scripts propios

from calcular_NCH_simple import *
from aux_functions import *

# #############################################################################
# Cargar librerías

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    AE_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/AE_results.txt")
    IF_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/IF_results.txt")
    LOF_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/LOF_results.txt")
    OCSVM_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/OCSVM_results.txt")
    RC_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/RC_results.txt")
    SVDD_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/SVDD_results.txt")
    NCH_results = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/NCH_results.txt")
    NCH_results2 = cargar_resultados_txt("C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Implementacion/Prueba1/resultados/NCH_results2.txt")

AE = parsear_y_calcular_metricas(AE_results)
IF = parsear_y_calcular_metricas(IF_results)
LOF = parsear_y_calcular_metricas(LOF_results)
OCSVM = parsear_y_calcular_metricas(OCSVM_results)
RC = parsear_y_calcular_metricas(RC_results)
SVDD = parsear_y_calcular_metricas(SVDD_results)
NCH = parsear_y_calcular_metricas(NCH_results)
NCH2 = parsear_y_calcular_metricas(NCH_results2)


indice = 3
print(obtener_mejor_metodo(AE, indice))
print(obtener_mejor_metodo(IF, indice))
print(obtener_mejor_metodo(LOF, indice))
print(obtener_mejor_metodo(OCSVM, indice))
print(obtener_mejor_metodo(RC, indice))
print(obtener_mejor_metodo(SVDD, indice))
print(obtener_mejor_metodo(NCH, indice))