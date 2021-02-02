# -*- coding: utf-8 -*-

# #############################################################################
# Cargar librerías

import math
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs, make_moons, make_s_curve
from scipy.spatial import Delaunay, ConvexHull
from IPython import get_ipython
from bentley_ottmann.planar import edges_intersect # Implementacion del algoritmo Bentley Ottmann

from aux_functions import *
# #############################################################################
def calcular_NCH_simple (args):
    X, l, extend, contraer_SCH = args
    # Delaunay tesselation of X
    #print("comenzando triangularizacion")
    tic = time.perf_counter()
    tri = Delaunay(X)
    toc = time.perf_counter()
    #print("tiempo triangularizacion: ", toc-tic)
    triangles = tri.simplices.copy()
    
    # Calcula el cierre convexo: aristas de borde del polígono
    #CH = tri.convex_hull # Antes usaba esto opción para el CH pero en la ayuda dice que no es recomendable por temas de inestabilidad
    #print("comenzando calculo CH")
    tic = time.perf_counter()
    CH   = ConvexHull(X)
    toc = time.perf_counter()
    #print("tiempo CH: ", toc-tic)
    triangles = tri.simplices.copy()
    CH_e = CH.simplices  # Aristas de borde del polígono
    
    # Cálculo de las longitudes de las aristas de borde
    dist = np.zeros(len(CH_e))
    j = 0
    for i in range(len(CH_e)):
            #plt.plot(X[:,0], X[:,1], 'ob')
            #plt.plot([X[CH[i,0],0],X[CH[i,1],0]],[X[CH[i,0],1],X[CH[i,1],1]],'r-')
            #plt.plot([X[CH[i,0],0],X[CH[i,1],0]],[X[CH[i,0],1],X[CH[i,1],1]],'ro')
            # Se calcula la distancia de cada arista del CH
            #plt.axis('equal')
            #plt.show()
    
            dist[j] = np.linalg.norm(X[CH_e[i,0]]-X[CH_e[i,1]])
            #print(j,dist[j])
            j = j + 1
    
    # Se ordenan de menor a mayor las aristas del cierre convexo en función de su longitud
    #print("comenzando ordenacion aristas")
    tic = time.perf_counter()
    index_sorted = np.argsort(dist)#[::-1]
    dist_sorted = np.sort(dist)#[::-1]
    toc = time.perf_counter()
    #print("tiempo ordenacion aristas: ", toc-tic)
    triangles = tri.simplices.copy()
    
    # Se crea una lista con las aristas de borde ordenadas por distancia (de menor a mayor)
    boundary_e = CH_e[index_sorted,:]
    
    # Se cread una lista con los vértices de borde
    #boundary_v = np.unique(CH_e) # ---> ¿¿¿Se podría cambiar por  ConvexHull(X).vertices?????. Creo que sí
    boundary_v = CH.vertices
    
    #1 plt.figure()
    #1 plt.plot(X[:,0], X[:,1], 'go')
    #1 plt.axis('equal')
    #1 plt.title('Data')
    #figManager = plt.get_current_fig_manager() 
    #figManager.window.showMaximized() # Maximiza la figura
    
    # Muestra la triangulazión inicial
    #1 plt.figure()
    #1 plt.triplot(X[:,0], X[:,1], tri.simplices.copy())
    #1 plt.plot(X[:,0], X[:,1], 'go')
    #1 plt.plot(X[boundary_v,0], X[boundary_v,1], 'ro', markersize=10)
    #1 plt.axis('equal')
    #1 plt.title('Delaunay triangulation')
    
    # Se crea un array vacío para contener las aristas del cierre no convexo final
    boundary_final = np.empty(shape=[0, 2],dtype=np.int32)
    
    #print("Comienzo poda aristas borde...")
    tic = time.perf_counter()
    while len(boundary_e)>0:
        edge = boundary_e[-1,:] # Se obtiene la arista de borde de mayor longitud
        dist_e = dist_sorted[-1] # Se obtiene la distancia de la arista de borde seleccionada
        boundary_e = np.delete(boundary_e, -1, 0) # Se elimina la última arista (la mayor por orden de longitud)
        dist_sorted = np.delete(dist_sorted,-1,0) # Se elimina la distancia corresponcdiente a la arista eliminada
        
        find_e = np.isin(triangles, edge) # Buscar los triángulos que contengan la arista 
        index = np.where(find_e) # Proporciona los índices de los triángulos que contienen algún vértice de la arista
        # Busca el índice duplicado que indicará el triángulo que contiene ambos vértices de la asista
        u, c = np.unique(index[0], return_counts=True)
        triangle_ix = u[c > 1]
        # Se obtiene el triángulo buscado (el que contiene la arista borde seleccionada)
        triangle_e = triangles[triangle_ix,:]
        # Se obtiene el tercer vértice del triángulo (es el que no está en la arista seleccionada)
        vertex = np.setdiff1d(triangle_e,edge)
        
        # Si la distancia de la arista es mayor que el umbral establecido y el tercer vértice no es de borde (para mantener regularidad)
        if (dist_e > l and not(vertex in boundary_v)):
            triangles = np.delete(triangles, triangle_ix, axis=0)  # Elimina el triángulo del polígono
            #print("triangle_ix", triangle_ix)
            #print("triangle_e", triangle_e)
            #print("vertex", vertex)
            #print("edge", edge)
            new_b_edge1 = [vertex[0], edge[0]] # Se obtiene la arista 1 del triángulo
            new_b_edge2 = [vertex[0], edge[1]] # Se obtiene la arista 2 del triángulo
            dist_edge1 = np.linalg.norm(X[new_b_edge1[0]]-X[new_b_edge1[1]]) # Calcula la longitud de arista 1
            dist_edge2 = np.linalg.norm(X[new_b_edge2[0]]-X[new_b_edge2[1]]) # Calcula la longitud de arista 2
            idx1 = np.searchsorted(dist_sorted,dist_edge1) # Busca la posición de la arista en la lista ordenada de longitudes
            boundary_e = np.insert(boundary_e,idx1,new_b_edge1,axis=0) # Inserta la arista 1 ordenada en el lista de aristas de borde
            dist_sorted = np.insert(dist_sorted,idx1,dist_edge1) # Inserta la longitud de arista 1 en la lista de longitudes
            idx2 = np.searchsorted(dist_sorted,dist_edge2) # Buscar la posición de la arista en la lista ordenada de longitudes
            boundary_e = np.insert(boundary_e,idx2,new_b_edge2,axis=0) # Inserta la arista 2 ordenada en el lista de aristas de borde
            dist_sorted = np.insert(dist_sorted,idx2,dist_edge2) # Inserta la longitud de arista 1 en la lista de longitudes
            
            #boundary_v = np.append(boundary_v,vertex) # Se añade el nuevo vértice en la lista de vértices de borde
            
            
            v1 = np.where(boundary_v == edge[0])[0][0]
            v2 = np.where(boundary_v == edge[1])[0][0]
            #print('Vértices ', v2, v1, edge, boundary_v, vertex)
            if (np.abs(v2-v1) > 1 and v1 != len(boundary_v)-1 and v2 != len(boundary_v)-1):
                print('Error!. Vértices no contiguos: ', v2, v1, edge, boundary_v)  
            if ((max(v2,v1) == len(boundary_v)-1) and ((min(v2,v1) == 0))):
                boundary_v = np.insert(boundary_v,0,vertex)
            else:
                boundary_v = np.insert(boundary_v,max(v2,v1),vertex)
        else:
            boundary_final = np.append(boundary_final,np.reshape(edge, (-1, 2)),axis=0)
    
    toc = time.perf_counter()
    #print("tiempo poda aristas borde: ", toc-tic)
    
    # Muestra la triangulazión final
    #1 plt.figure()
    #1 plt.triplot(X[:,0], X[:,1], triangles)
    #1 plt.plot(X[:,0], X[:,1], 'go')
    #1 plt.plot(X[boundary_v,0], X[boundary_v,1], 'ro', markersize=10)
    #1 plt.axis('equal')
    #1 plt.title('Non-convex clousure with l=%s' %l)
    
    # Muestra el borde del cierre no convexo final
    #1 plt.plot([X[boundary_final[:,0],0],X[boundary_final[:,1],0]],[X[boundary_final[:,0],1],X[boundary_final[:,1],1]],'r-')
    
    #print("comenzando parte final")
    tic = time.perf_counter()
    
    if (contraer_SCH == True):
        extend_list = np.arange(extend, 0, -extend/5)
        extend_list = np.append(extend_list, 0)
    else:
        extend_list = np.array([extend])
        extend_list = np.append(extend_list, 0)
        
    for e in extend_list: 
        # 2 print()
        # 2 print("extend: ", e)
        # Recorre los vértices externos del polígono final
        if (e != 0):
            
            sign_ang = []
            incenter_l = np.empty(shape=[0, 2])
            extVertex_l = np.empty(shape=[0, 2])
            z = 0
            count = 0
            Xordenado = np.zeros(X.shape)
            
            for i in boundary_v:
                find_v = np.isin(triangles, i)  # Busca los triángulos que contengan ese vértice
                index_t = np.where(find_v)      # Localiza las posiciones de los triángulos
                index_t = index_t[0]            # Se queda con el primer índice ya que indica el número de triángulo
                sum_angle = 0
                # Recorre los triángulos seleccioandos para calcular el ángulo interior del vértice externos
                for j in index_t:              
                    vertices = np.setdiff1d(triangles[j,:],i)   # Obtiene los otros vértices del triángulo que no son el seleccionado
                    a = np.linalg.norm(X[vertices[0]]-X[i])     # Calcula la longitud del primer lado del triángulo
                    b = np.linalg.norm(X[vertices[1]]-X[i])     # Calcula la longitud del segundo lado del triángulo
                    c = np.linalg.norm(X[vertices[0]]-X[vertices[1]]) # Calcula la longitud del tercer lado del triángulo
                    #print("a :", a)
                    #print("b :", b)
                    #print("c :", c)
                    aux_ang = ( a**2 + b**2 - c**2 ) / (2*a*b)
                    if aux_ang > 1:
                        aux_ang = 1
                    elif aux_ang < -1:
                        aux_ang = -1
                        
                    #print("aux_ang: ", aux_ang)
                    angle = np.degrees ( math.acos( aux_ang ) ) # Cácula el ángulo para el vértice dado
                    sum_angle = sum_angle + angle
                    
                # Cálculos previos para determinar el vértice extendido a partir del vértice externo
                find_e = np.isin(boundary_final, i)
                index_e = np.where(find_e)
                index_e = index_e[0]
                edges = boundary_final[index_e]
                points = np.setdiff1d(edges,i)
                edges = np.append(edges,[points],axis=0)
                lenEdges = np.linalg.norm(X[edges[:,0]]-X[edges[:,1]],axis=1)        
                incenter = (X[i,:]*lenEdges[2]+X[np.setdiff1d(edges[0],i),:][0]*lenEdges[1]+X[np.setdiff1d(edges[1],i),:][0]*lenEdges[0])/ sum(lenEdges) # Basado en: https://es.wikipedia.org/wiki/Incentro
                lenAB = np.linalg.norm(X[i]-incenter) 
                    
                # Indica si el vértice es cóncavo a convexo (función de la suma de todos los ángulos de los triángulos)
                if sum_angle>180:
                    # checkpoint plt.text(X[i,0],X[i,1],'Concave',fontsize=14,fontweight='bold')  
                    sign_ang = np.append(sign_ang,-1) # Si el ángulo es cóncava se restará sobre el vértice externo
                else:
                    # checkpoint plt.text(X[i,0],X[i,1],'Convex',fontsize=14,fontweight='bold')   
                    sign_ang = np.append(sign_ang,1) # Si el ángulo es convexo se sumará sobre el vértice externo
                    
                # Calcula el vértice extendido en función de si es cóncavo o convexo (valor de sign_ang)
                #print("X[i]: ", X[i])
                #print("sign_ang[z]: ", sign_ang[z])
                #print("incenter: ", incenter)
                #print("lenAB: ", lenAB)
                #print("e: ", e)
                extVertex = X[i] + sign_ang[z] * (X[i] - incenter) / lenAB * e # Basado en: https://stackoverflow.com/questions/7740507/extend-a-line-segment-a-specific-distance
                z = z + 1
                extVertex_l = np.append(extVertex_l, np.reshape(extVertex, (-1, 2)),axis=0) 
                # Almacenamos el vértice extendido
                
                
                # Dibuja el vértice extendido y el incentro usado para calcularlo
                # checkpoint plt.plot(incenter[0],incenter[1],'mo')
                #plt.plot(extVertex[0],extVertex[1],'bo')
                #plt.text(extVertex[0],extVertex[1],'v_'+str(z),fontsize=14,fontweight='bold')
            
            # Comprobar si el SNCH es simple o complejo
            # 2 print("Complejo: ",edges_intersect(array_to_sequence_of_vertices(extVertex_l)))
            #print("comenzando comprobacion poligono simple/complejo")
            tic1 = time.perf_counter()
            if (edges_intersect(array_to_sequence_of_vertices(extVertex_l)) == False):
                extend = e
                break
            toc1 = time.perf_counter()
            #print("tiempo poda aristas borde: ", toc1-tic1)
        elif (e == 0):
            extend = 0
            extVertex_l = X
    
    
    toc = time.perf_counter()
    #print("tiempo parte final: ", toc-tic)

    #    plt.plot(incenter_l[z,0],incenter_l[z,1],'yo', markersize=12)
    #    plt.plot(X[v,0],X[v,1],'yo')    
    #    plt.plot(extVertex[:,0],extVertex[:,1],'yx',markersize=12)
        
        #plt.plot(extVertex1[0],extVertex[1],'yx')
        
    #    z = np.where(np.isin(boundary_v, i))
    #    lenAB = np.linalg.norm(X[i]-incenter_l[z],axis=0) 
    #    extVertex = X[i] + sign_ang[z] * (X[i] - incenter_l[z]) / lenAB * extend 
    #    
    #    plt.plot(incenter_l[z,0],incenter_l[z,1],'yo')
    #    plt.plot(X[i,0],X[i,1],'yo')
    #    plt.plot(extVertex[:,0],extVertex[:,1],'yx')
    #    
    #    plt.plot(extVertex[:,0],extVertex[:,1],'r--')
    
    return X, boundary_final, extVertex_l , boundary_v, extend
