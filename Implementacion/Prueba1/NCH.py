# -*- coding: utf-8 -*-

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
from IPython import get_ipython 
from sklearn.datasets.samples_generator import make_blobs, make_moons, make_s_curve # Generar datasets artificiales
from scipy.spatial import Delaunay, ConvexHull # Triangulizacion de Delaunay y calculo del Convex Hull
from bentley_ottmann.planar import edges_intersect # Implementacion del algoritmo Bentley Ottmann

# #############################################################################
# Generar datasets artificiales

ndat = 5000     # Tamaño del conjunto de datos
dataset = 1     # Indica el data set elegido

# Seleccionar dataset
random.seed(10)
if dataset == 1:
    X, y = make_blobs(n_samples=ndat, centers=2, n_features=2, cluster_std=0.5, random_state=0)
elif dataset==2:
    X, y = make_moons(n_samples=ndat*2, noise=0.05)
    X = X[y==1] # Seleccionamos datos únicamente de una media luna
    y = y[y==1]
elif dataset==3:
    X, y = make_s_curve(n_samples=ndat, noise=0.1, random_state=0)
    X = X[:,[0,2]]

# #############################################################################
# Probar algoritmo

# Indica que las figuras se mostrarán en ventanas independientes
get_ipython().run_line_magic('matplotlib', 'qt')
# Cierra todas las figuras abiertas
plt.close('all')

l = 0.3         # Hiperparámetro del modelo (distancia mínima de las aristas)
extend = .7   # Indica la longitud en que se extiende cada vértice del cierre no convexo

vertices, aristas, vertices_expandidos, vertices_ordenados = calcular_NCH_simple(X, l, extend)

print("Intersección vértices internos: ", edges_intersect(array_to_sequence_of_vertices(vertices[vertices_ordenados])))
print("Intersección vértices expandidos: ",edges_intersect(array_to_sequence_of_vertices(vertices_expandidos)))

for i in range(len(vertices[vertices_ordenados])):
    plt.text(X[vertices_ordenados[i]][0],X[vertices_ordenados[i]][1],i)

plt.xlim(-5, 5)
plt.ylim(-5, 5)



