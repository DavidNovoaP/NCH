# -*- coding: utf-8 -*-

# #############################################################################
# Cargar scripts propios

from calcular_NCH_simple import *
from aux_functions import *
from Algorithms import *
#from calcular_NCH_simple import *
from NCH_parallel import *
#from NCH import *
import io


# #############################################################################
# Cargar librerías

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import get_ipython 
from sklearn.datasets.samples_generator import make_blobs, make_moons, make_s_curve # Generar datasets artificiales
from scipy.spatial import Delaunay, ConvexHull # Triangulizacion de Delaunay y calculo del Convex Hull
from bentley_ottmann.planar import edges_intersect # Implementacion del algoritmo Bentley Ottmann


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import multiprocessing as mp
import time
from sklearn import svm

from pathos.multiprocessing import ProcessingPool as Pool



if __name__ == '__main__':
    
    # #############################################################################
    # Generar datasets artificiales
    """
    ndat = 5     # Tamaño del conjunto de datos
    dataset = 1     # Indica el data set elegido
    
    # Seleccionar dataset
    random.seed(10)
    if dataset == 1:
        X, y = make_blobs(n_samples=ndat, centers=2, n_features=3, cluster_std=0.5, random_state=0)
    elif dataset==2:
        X, y = make_moons(n_samples=ndat*2, noise=0.05)
        X = X[y==1] # Seleccionamos datos únicamente de una media luna
        y = y[y==1]
    elif dataset==3:
        X, y = make_s_curve(n_samples=ndat, noise=0.1, random_state=0)
        X = X[:,[0,2]]
    """
    

    # #############################################################################
    # Telescope
    """
    path = "C:/Users/DAVID/Desktop/TESIS/NCH/Datasets/MagicTelescope11/magic04.txt"
    #path = "C:/Users/usuario/Desktop/LIDIA/TESIS/NCH/Datasets/MagicTelescope11/magic04.txt"
    dataset = pd.read_csv(path)
    dataset = dataset.drop_duplicates()
    dataset = dataset.reset_index(drop = True)
    
    X = dataset.iloc[:, 0:dataset.shape[1]-1]
    Y = dataset.iloc[:, dataset.shape[1]-1]
    
    # For technical reasons, the number of h events is underestimated. In the real data, the h class represents the majority of the events.
    # 'g' = gamma (signal) = normal class
    # 'h' = hadron (background) = anomaly class
    Y = Y.apply(change_target_value_GH)
    
    """
    # #############################################################################
    # MinibooNE
    """
    path = "C:/Users/DAVID/Desktop/TESIS/Proyectos/NCH/Datasets/Miniboone/MiniBooNE_PID.txt" # 
    dataset = []
    with io.open(path, mode="r", encoding="utf-8") as f:
        next(f)
        for line in f:
            dataset.append(map(float, line.split()))
    dataset = pd.DataFrame(dataset)
    dataset.insert(dataset.shape[1], "Class", -1)
    dataset.iloc[0:36499, dataset.shape[1]-1] = 1
    dataset.iloc[36499:dataset.shape[0], dataset.shape[1]-1] = 0
    dataset.columns = dataset.columns.astype(str)
    dataset = dataset.drop_duplicates()
    dataset = dataset.reset_index(drop = True)
    X = dataset.iloc[:, 0:dataset.shape[1]-1]
    Y = dataset.iloc[:, dataset.shape[1]-1]
    
    """
    # #############################################################################
    # MNIST
    """
    from mnist import MNIST
    path_mnist = "C:/Users/DAVID/Desktop/TESIS/Proyectos/NCH/Datasets/MNIST/"
    mndata = MNIST(path_mnist)
    X_train, Y_train = mndata.load_training()
    X_test, Y_test = mndata.load_testing()
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    Y_train = pd.Series(Y_train)
    Y_test = pd.Series(Y_test)
    
    Y_train = Y_train.apply(change_target_value_MNIST)
    Y_test = Y_test.apply(change_target_value_MNIST)
    
    normal_data_indexes = Y_train.index[Y_train == 0].tolist()
    X_train = X_train.iloc[normal_data_indexes, :]
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)
    
    X_train = X_train.drop_duplicates()
    X_train = X_train.reset_index(drop = True)
    X_train = X_train.sample(frac = 1) 
    
    #model_normalizer = NormalizeData_Train(X_train)
    #X_train = NormalizeData(X_train, model_normalizer)
    #X_test = NormalizeData(X_test, model_normalizer)
    
    X = X_train.iloc[0:20000, :]
    Y = Y_train.iloc[0:20000]
    
    
    # #############################################################################
    # MAMOGRAPHY
    
    import scipy.io
    path_mamo = "C:/Users/DAVID/Desktop/TESIS/Proyectos/NCH/Datasets/"
    mat = scipy.io.loadmat(path_mamo + 'mammography.mat')
    
    X = pd.DataFrame(mat['X'])
    X.insert(X.shape[1], "Class", pd.DataFrame(mat['y'].flatten()))
    X = X.drop_duplicates()
    X = X.reset_index(drop = True)
    Y = pd.Series(X.loc[:, "Class"])
    X = X.iloc[:, 0:X.shape[1]-1]
    """
    # #############################################################################
    # SHUTTLE
    
    import scipy.io
    path_mamo = "C:/Users/DAVID/Desktop/TESIS/Proyectos/NCH/Datasets/"
    mat = scipy.io.loadmat(path_mamo + 'shuttle.mat')
    
    X = pd.DataFrame(mat['X'])
    X.insert(X.shape[1], "Class", pd.DataFrame(mat['y'].flatten()))
    
    X = X.drop_duplicates()
    X = X.reset_index(drop = True)
    Y = pd.Series(X.loc[:, "Class"])
    X = X.iloc[:, 0:X.shape[1]-1]
    
    # #############################################################################
    # Splitear datos
    
    normal_data_indexes = Y.index[Y == 0].tolist()
    anomaly_data_indexes = Y.index[Y == 1].tolist()
    
    random.shuffle(normal_data_indexes)
    random.shuffle(anomaly_data_indexes)
    
    normal_data_indexes = normal_data_indexes[0:10000] # en el caso de MINIBOONE
    
    NCH_results_g = []
    RC_results_g = []
    OCSVM_results_g = []
    LOF_results_g = []
    IF_results_g = []
    AE_results_g = []
    SVDD_results_g = []
    
    splits = 10
    for split in range (0, splits):
        
        train_normal_data_indexes, test_normal_data_indexes = train_test_split(normal_data_indexes, test_size=0.1, random_state=split) # mammo
        
        anomaly_data_indexes = anomaly_data_indexes[0:len(test_normal_data_indexes)] # MINIBOONE
        test_normal_data_indexes = test_normal_data_indexes[0:len(anomaly_data_indexes)] # mammo
        
        X_train = X.iloc[train_normal_data_indexes, :]
        Y_train = Y.iloc[train_normal_data_indexes]
        
        X_test = X.iloc[test_normal_data_indexes + anomaly_data_indexes, :]
        Y_test = Y.iloc[test_normal_data_indexes + anomaly_data_indexes]
    
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)
    
        # #############################################################################
        # Non Convex Hull
        """
        
        l = 20000       # Hiperparámetro del modelo, distancia mínima de las aristas (más L => menos ajustado)
        extend = 0.0001   # Indica la longitud en que se extiende cada vértice del cierre no convexo
        n_proy = 2000 # Número de proyecciones a emplear
        threads = 10     # Número de procesadores a emplear en el caso de multiproceso
    
        # Entrenar    
        model = NCH_train (X_train.to_numpy(), n_proy, l, extend, False)
        result = NCH_classify (X_test.to_numpy(), model)
        
        # Evaluar resultados 
        titulo = "-L: "+str(l) + ", Extend: " + str(extend) + ", Proyecciones: " + str(n_proy)
        calcular_metricas(Y_test, result, titulo)
        
        
        
        NCH_parameters1 = [2] # Proyecciones
        NCH_parameters2 = [0.6, 0.55, 0.50, 0.45, 0.40, 0.35, 0.32] # l
        NCH_parameters3 = [0.0005, 0.001, 0.0015, 0.0025, 0.0035, 0.005, 0.006] # extend
        NCH_parameters4 = [8] # threads
        NCH_results = []
        """
        print("k-fold: ", split)
        NCH_parameters1 = [2000] # Proyecciones
        NCH_parameters2 = [20000] #[0.2, 0.3, 0.45, 0.55, 0.7, 0.9] # l
        #NCH_parameters2 = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.2] # l
        NCH_parameters3 = [0.00001, 0.001, 0.1] #, 0.0005, 0.001, 0.0015, 0.0025, 0.0035, 0.005, 0.006] # extend
        NCH_parameters4 = [8] # threads
        NCH_results = []
        
        process_pool = Pool(nodes=NCH_parameters4[0])
        
        for i in NCH_parameters1:
            for j in NCH_parameters2:
                for k in NCH_parameters3:
                    for l in NCH_parameters4:
                        tic_train = time.perf_counter() 
                        NCH_model = NCH_train (X_train.to_numpy(), i, j, k, False, process_pool)
                        toc_train = time.perf_counter()
                        
                        tic_test = time.perf_counter() 
                        NCH_predict = NCH_classify (X_test.to_numpy(), NCH_model, process_pool)
                        toc_test = time.perf_counter()
                        
                        titulo = "NCH -Proyecciones: " + str(i) + " -l: " + str(j) + " -Extend: " + str(k) + " -Threads: " + str(l) + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
                        cm = calcular_metricas(Y_test, NCH_predict, titulo)
                        cm.append(titulo)
                        NCH_results.append(cm)
        NCH_results_g.append(NCH_results)
        
        process_pool.close()
        process_pool.join()
        
        process_pool.restart()
        
        
        # #############################################################################
        # Robust Covariance
        print("k-fold: ", split)
        RC_parameters1 = [ 0.05, 0.1, 0.2, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]# Contamination
        RC_results = []
        for i in RC_parameters1:
            tic_train = time.perf_counter() 
            RC_model = RC_train(X_train.to_numpy(),i)
            toc_train = time.perf_counter()
            
            tic_test = time.perf_counter() 
            RC_predict = sk_classify(X_test, RC_model)
            toc_test = time.perf_counter()
            
            titulo = "RC -Contamination: " + str(i) + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
            cm = calcular_metricas(Y_test, RC_predict, titulo)
            cm.append(titulo)
            RC_results.append(cm)
        RC_results_g.append(RC_results)
        
        # #############################################################################
        # One Class SVM
        print("k-fold: ", split)
        OCSVM_parameters1 = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9] # nu == contamination
        OCSVM_parameters2 = ["linear"]#, "poly", "rbf", "sigmoid"] # kernel
        OCSVM_parameters3 = ["auto", "scale"] # gamma
        OCSVM_results = []
        for i in OCSVM_parameters1:
            for j in OCSVM_parameters2:
                for k in OCSVM_parameters3:
                    
                    tic_train = time.perf_counter()
                    OCSVM_model = OCSVM_train(X_train.to_numpy(),i, j, k)
                    toc_train = time.perf_counter()
                    
                    tic_test = time.perf_counter()
                    OCSVM_predict = sk_classify(X_test, OCSVM_model)
                    toc_test = time.perf_counter()
                    
                    titulo = "OCSVM -Nu: " + str(i) + " -Kernel: "+ j + " -Gamma: " + k  + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
                    cm = calcular_metricas(Y_test, OCSVM_predict, titulo)
                    cm.append(titulo)
                    OCSVM_results.append(cm)
        OCSVM_results_g.append(OCSVM_results)
        
        # #############################################################################
        # Isolation Forest
        print("k-fold: ", split)
        IF_parameters1 = [5, 15, 50, 100, 200] # n_estimators
        IF_parameters2 = [0.01, 0.05, 0.1, 0.2] # contamination
        IF_parameters3 = [7, 11, 13] # random_state
        IF_results = []
        
        for i in IF_parameters1:
            for j in IF_parameters2:
                for k in IF_parameters3:
                    
                    tic_train = time.perf_counter()
                    IF_model = IF_train(X_train.to_numpy(),i, j, k)
                    toc_train = time.perf_counter()
                    
                    tic_test = time.perf_counter()
                    IF_predict = sk_classify(X_test, IF_model)
                    toc_test = time.perf_counter()
                    
                    titulo = "IF -m_estimators: " + str(i) + " -Contamination: " + str(j) + " -Random state: " + str(k) + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
                    cm = calcular_metricas(Y_test, IF_predict, titulo)
                    cm.append(titulo)
                    IF_results.append(cm)
        IF_results_g.append(IF_results)
        
        # #############################################################################
        # Local Outlier Factor
        print("k-fold: ", split)
        LOF_parameters1 = [15, 50, 100] # n_neighbors
        LOF_parameters2 = [0.1, 0.2, 0.35, 0.4, 0.45, 0.5] # contamination
        LOF_results = []
        
        for i in LOF_parameters1:
            for j in LOF_parameters2:
                    
                    tic_train = time.perf_counter()
                    LOF_model = LOF_train(X_train.to_numpy(),i, j, "True", "auto")
                    toc_train = time.perf_counter()
                    
                    tic_test = time.perf_counter()
                    LOF_predict = sk_classify(X_test, LOF_model)
                    toc_test = time.perf_counter()
                    
                    titulo = "LOF -n_neighbors: " + str(i) + " -Contamination: " + str(j) + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
                    cm = calcular_metricas(Y_test, LOF_predict, titulo)
                    cm.append(titulo)
                    LOF_results.append(cm)
        LOF_results_g.append(LOF_results)
        
        # #############################################################################
        # Autoencoder
        print("k-fold: ", split)
        AE_parameters1 = [[10,5,10], [10,10,8, 10, 10], [5,2,5], [6,4,6], [10, 8, 6, 8, 10]] # hidden layers
        AE_parameters2 = [100, 500, 1000] # epochs
        AE_parameters3 = [ 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1] # contamination
        AE_results = []
        
        for i in AE_parameters1:
            for j in AE_parameters2:
                for k in AE_parameters3:
                    
                    tic_train = time.perf_counter()
                    AE_model = AE_train(X_train, i, j, k)
                    toc_train = time.perf_counter()
                    
                    tic_test = time.perf_counter()
                    AE_predict = AE_classify(X_test, AE_model)
                    toc_test = time.perf_counter()
                    
                    titulo = "AE -hidden layers: " + str(i) + " -Epochs: " + str(j) + " -Contamination: " + str(k) + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
                    cm = calcular_metricas(Y_test, AE_predict, titulo)
                    cm.append(titulo)
                    AE_results.append(cm)              
        AE_results_g.append(AE_results)
    
    """  
        
        # #############################################################################
        # SVDD
        print("k-fold: ", split)
        SVDD_parameters1 = [0.1, 0.4, 0.9] # positive_penalty
        SVDD_parameters2 = [0.1, 0.4, 0.9] # negative_penalty
        SVDD_parameters3 = ["1", "2", "3", "4"] # kernel
        SVDD_results = []
    
        
        for i in SVDD_parameters1:
            for j in SVDD_parameters2:
                for k in SVDD_parameters3:
                    
                    tic_train = time.perf_counter()
                    SVDD_model = SVDD_train(np.array(X_train), np.array(pd.DataFrame(Y_train)), i, j, k)
                    toc_train = time.perf_counter()
                    
                    tic_test = time.perf_counter()
                    SVDD_predict = SVDD_classify(np.array(X_test), np.array(pd.DataFrame(Y_test)), SVDD_model)
                    toc_test = time.perf_counter()
                    
                    titulo = "SVDD -positive_penalty: " + str(i) + " -negative_penalty: " + str(j) + " -kernel: " + str(k) + " -Tiempo train/test: "+ str(toc_train-tic_train) + " "+ str(toc_test-tic_test)
                    cm = calcular_metricas(Y_test, SVDD_predict, titulo)
                    cm.append(titulo)
                    SVDD_results.append(cm)  
        SVDD_results_g.append(SVDD_results)      
       
    # #############################################################################
    with open("SVDD_results.txt", "w") as output:
        output.write(str(SVDD_results))
    """

    
    
