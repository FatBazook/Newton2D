# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:52:09 2022

@author: vpinardon
"""

import numpy as np

def newton(F,DF,x0, eps=1e-4, N=100):
    x = x0.copy()
    for i in range(N):
        Fx = F(x)
        DFx = DF(x)
        if np.linalg.norm(Fx) < eps:
            return x
        if np.linalg.norm(DFx) < eps:
            raise Exception (f"la derivée |DF| = {np.linalg.norm(DFx)} est trop petite")
        x -= np.linalg.solve(DFx,Fx)
    raise Exception(f"l'erreur après {N} itérations est {np.linalg.norm(Fx)} > {eps}")
    

F = lambda x: np.array([np.cos(x[0]) - np.sin(x[1]),
                        np.exp(-x[0]) - np.cos(x[1])])

DF = lambda x: np.array([[-np.sin(x[0]), -np.cos(x[1])],
                         [-np.exp(x[0]), np.sin(x[1])]])

newton(F,DF, np.array([0.,0.]))