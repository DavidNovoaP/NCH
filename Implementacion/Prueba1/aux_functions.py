# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 19:25:52 2020

@author: DAVID
"""

def array_to_sequence_of_vertices (data):
    aux_list = []
    for i in range (0, data.shape[0]):
        aux_list.append((data[i, 0],data[i, 1]))
        
    return aux_list