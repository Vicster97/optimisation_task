# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:53:47 2020

@author: veronika ulanova
"""

# load usual python packages
import numpy as np
import scipy as sp
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import math 

#load optimisation packages
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import fmin_bfgs

# for reading and displaying images
#from skimage.io import imread
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

#to allow argument input from cmd pannel 
import argparse


#take arguments from cmd 
parser = argparse.ArgumentParser()
parser.add_argument("p_arg")
parser.add_argument("r_arg")
args = parser.parse_args()

#apply exceptions if the input parameters are out of range
if (float(args.p_arg) <-2) or (float(args.p_arg) >2):
    raise Exception("The input value of the p parameter is out of range. Try "
                    "choosing a value between -2 and 2.")
else:
    p = float(args.p_arg) #otherwise accept the parameters 
    
if (float(args.r_arg) <10) or (float(args.r_arg) >50):
    raise Exception("The input value of the r parameter is out of range. Try "
                    "choosing a value between 10 and 50.")
else:
    r = float(args.r_arg)

# define the functions that we are minimising 
def L_fn_pd(coeff, x_j, p_arg,r_arg):
    """ Calculates the value of L.
    
    Parameters: 
    coeff (array): 1D array of a_i,b_i,c_i and d_i parameters 
    with shape (2000)
    x_j (array or list): list of x_j values 
    p_arg (float): p parameter
    r_arg (float): r parameter
        
    Returns:
    float: value of the function
    
    """
    coefficients = pd.DataFrame(
            data=coeff.reshape(500,4),
            columns=['a','b','c','d']
            )
    
    x_param = pd.DataFrame(
            data=x_j,
            columns=['x_j']
            )
    
    f = float(
            x_param.applymap(
                    lambda x:((coefficients['d']
                    +coefficients['c']*np.tanh(coefficients['a']*x
                                 +coefficients['b'])).sum()
    - p_arg*np.cos(r_arg*x))**2).sum())

    return f

#define the function that calculates the jacobians
def L_jac_pd(coeff, x_j, p_arg,r_arg):
    """ Calculates the jacobians for the L function.
    
    Parameters: 
    coeff (array): 1D array of a_i,b_i,c_i and d_i parameters 
    with shape (2000)
    x_j (array or list): list of x_j values 
    p_arg (float): p parameter
    r_arg (float): r parameter
        
    Returns:
    tulpe: jacobian (i.e. raw vector)
    
    """
    
    coefficients = pd.DataFrame(
            data=coeff.reshape(500,4),
            columns=['a','b','c','d']
            )
    
    x_param = pd.DataFrame(
            data=x_j,
            columns=['x_j']
            )
    
    #getting a_i jacobians
    coefficients['df_a_i'] = (
            x_param.applymap(lambda x: (
                    2*((coefficients['d']
                    +coefficients['c']
                    *np.tanh(coefficients['a']*x
                             +coefficients['b'])).sum()
    - p_arg*np.cos(r_arg*x))
    *(coefficients['c']*x*(
            1-np.tanh(coefficients['a']*x
                      +coefficients['b']))**2))).sum()).sum()
    
    #getting b_i jacobians
    coefficients['df_b_i'] = (
            x_param.applymap(lambda x:2*((
                    coefficients['d']
                    +coefficients['c']
                    *np.tanh(coefficients['a']*x
                             +coefficients['b'])).sum()
    - p_arg*np.cos(r_arg*x))
    *(coefficients['c']*(1-np.tanh(coefficients['a']*x
      +coefficients['b']))**2)).sum()).sum()
    
    #getting c_i jacobians
    coefficients['df_c_i'] = (
            x_param.applymap(lambda x:2*((
                    coefficients['d']
                    +coefficients['c']
                    *np.tanh(coefficients['a']*x
                             +coefficients['b'])).sum()
    - p_arg*np.cos(r_arg*x))
    *( np.tanh(coefficients['a']*x
               +coefficients['b']))).sum()).sum()
    
    #getting d_i jacobians
    coefficients['df_d_i'] = (
            x_param.applymap(lambda x: 2*((
                    coefficients['d']
                    +coefficients['c']
                    *np.tanh(coefficients['a']*x
                             +coefficients['b'])).sum()
    - p_arg*np.cos(r_arg*x))).sum()).sum()

    jac_all = np.array(coefficients.iloc[:,-4:]).reshape(2000)
    return jac_all

print ("Both parameteres are within range. Starting optimisation.")

# defining the x_j parameter
j = np.linspace(1, 100, 100)
x_j = (j-1)/99

#choose the initial values (initial guess)
init = np.zeros(2000)

#run optimisation
sols_7 = fmin_bfgs(L_fn_pd, init, L_jac_pd, args=(x_j,p,r))

#outputting the parameter values
np.savetxt("out",sols_7.reshape(500,4))
print("Output parameters saved.")