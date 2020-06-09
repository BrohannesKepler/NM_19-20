#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 20:13:54 2019

@author: haider
"""


import numpy as np
from matplotlib import pyplot as plt



def rk3(A, bvector, y0, interval, N):
    
    """ 
    Functional implementation of the standard explicit third order
    Runge-Kutta (RK3) method. This will solve equations of the form:
        
        dydt = Ay + b(x)                                            (1)
        
            
    A is an (nxn) matrix with constant coefficients and b depends
    on the indpendent variable x only.
    
    This function uses standard Numpy routines for matrix multiplications
        
    
    Parameters
    ----------
    
        Requires the following inputs:
        
    A:  array
        The matrix A as defined in equation (1)
    
    bvector: function
             A function in which the input is the location x and the 
             output is the vector b in equation (1)
    
    y0: array
        n-vector of initial data for the system
    
    interval: list
              A list interval which provides the start and end values
              on which the solution is computed [x0, xn]
              
    N:  int
        Number of steps the algorithm will take, defined so that 
        h = (xn-x0)/N 
           
    
    Returns
    -------
    
    x:  array
        Vector of locations at which the solution has been evaluated, uniformly
        spaced.
        
    y:  array
        Vector of solutions to the problem, with dimensions MxN where M is 
        the number of equations in the system and N is the number of solution 
        steps (length of x)/
    
    The solver will output vector x containing the locations x at 
    which the solution is evaluated, and a vector y containing the solutions.


              
    """
               
    
    #Check inputs will not violate the programme:
    
    #Check if A is an array:
    
    assert(isinstance(A, np.ndarray)), 'A must be an array'
    
    #Check that A is a square matrix:
    
    assert(A.shape[0] == A.shape[1]), 'The matrix A must be square'
    
    #Check that the number of initial conditions is sufficient:
    
    assert(len(y0) == len(A)), 'Incorrect number of initial conditions'
    
    #Check that the number of steps in the algorithm is an integer:
    
    assert(isinstance(N, int)), 'N must be an integer'
    
    #Check that bvector is a function:
    
    assert(callable(bvector)), 'bvector must be a defined function'
    
    #Check that interval is a suitable iterable of two elements:
    
    assert(len(interval) == 2), 'interval must be a list of two elements'
    
    #Check the elements in interval are not equal:
    
    assert(interval[0] != interval[1]), 'interval must have two different elements'
    
        
    #Define the nodes of x:
    
    x = np.linspace(interval[0], interval[1], N+1)
    
    #Compute the step-size for the solution:
    
    h = (interval[1] - interval[0])/N
    
    #Preallocate the solution vector as an array of zeros for efficiency
    #when solving:
    
    Y = np.zeros((A.shape[0], len(x)))
    Y[:, 0] = y0
    
    #Loop through the range of the location vector (minus 1) as we already 
    #have the initial value
    
    for i in range(0, N):
        
        #Compute Y(1) and Y(2):
        
        Y1  = Y[:, i] + h*(np.matmul(A, Y[:, i]) + bvector(x[i]))
        Y2 = 0.75*Y[:, i] + 0.25*Y1 + 0.25*h*(np.matmul(A, Y1) + bvector(x[i] + h))
        
        #Now compute the weighted sum of the solution vector Y at step (n+1)
        
        Y[:, i+1] = (1/3)*Y[:, i] + (2/3)*Y2 + (2/3)*h*(np.matmul(A, Y2) + bvector(x[i] + h))
        
    
    return x, Y
        
    
    
def dirk3(A, bvector, y0, interval, N):
    
    """ 
    Functional representation of the two stage third order accurate 
    Diagonally Implicit Runge-Kutta method (DIRK3). This will implicitly
    solve a system of equations defined by:
        
        dydt = Ay + b(x)                                            (1)
        
    A is an (nxn) matrix with constant coefficients and b depends
    on the indpendent variable x only
    
    This function uses standard numpy routines to solve for the linear systems
    for y1 and y2, as well as for matrix multiplication.
        
        
    Parameters
    ----------
        
    A:  array
        The matrix A as defined in equation (1)
    
    bvector: function
             A function in which the input is the location x and the 
             output is the vector b in equation (1)
    
    y0: array
        n-vector of initial data for the system
    
    interval: list
              A list interval which provides the start and end values
              on which the solution is computed [x0, xn]
              
    N:  int
        Number of steps the algorithm will take, defined so that 
        h = (xn-x0)/N 
           
    
    Returns
    -------
    
    x:  array
        Vector of locations at which the solution has been evaluated, uniformly
        spaced.
        
    y:  array
        Vector of solutions to the problem, with dimensions MxN where M is 
        the number of equations in the system and N is the number of solution 
        steps (length of x)/
    
    The solver will output vector x containing the locations x at 
    which the solution is evaluated, and a vector y containing the solutions.

    """

    #Check inputs will not violate the programme:
    
    #Check if A is an array:
    
    assert(isinstance(A, np.ndarray)), 'A must be an array'
    
    #Check that A is a square matrix:
    
    assert(A.shape[0] == A.shape[1]), 'The matrix A must be square'
    
    #Check that the number of initial conditions is sufficient:
    
    assert(len(y0) == len(A)), 'Incorrect number of initial conditions'
    
    #Check that the number of steps in the algorithm is an integer:
    
    assert(isinstance(N, int)), 'N must be an integer'
    
    #Check that bvector is a function:
    
    assert(callable(bvector)), 'bvector must be a defined function'
    
    #Check that interval is a suitable iterable of two elements:
    
    assert(len(interval) == 2), 'interval must be a list of two elements'
    
    #Check the elements in interval are not equal:
    
    assert(interval[0] != interval[1]), 'interval must have two different elements'
    

    
    #Define the nodes of x:
    
    x = np.linspace(interval[0], interval[1], N+1)
    
    #Compute the step-size for the solution:
    
    h = (interval[1] - interval[0])/N

    #Preallocate the solution vector as an array of zeros for efficiency
    #when solving:
    
    Y = np.zeros((A.shape[0], len(x)))
    Y[:, 0] = y0
    
    
    #Defining the coefficients prior to loop for efficiency:
    
    mu = 0.5*(1 - (1/np.sqrt(3)))
    nu = 0.5*(np.sqrt(3) - 1)
    gm = 1.5 / (3 + np.sqrt(3))
    lm = 1.5 * ((1 + np.sqrt(3))/(3 + np.sqrt(3)))
    
    #Loop through the range of the location vector (minus 1) as we already 
    #have the initial value
    
    for i in range(0, N):

        #Define a C matrix which will be used to solve Cy(1/2)=b, the secondary 
        #system within the solution
        
        c = np.identity(A.shape[0]) - (h*mu*A)
        
        Y1 = np.linalg.solve(c, (Y[:, i] + h*mu*bvector(x[i] + h*mu)))

        #Define a b matrix for the Y2 system for ease:
        
        by2 = Y1 + h*nu*(np.matmul(A, Y1) + bvector(x[i] + h*mu)) + h*mu*bvector(x[i] + h*nu + 2*h*mu)
    
        Y2 = np.linalg.solve(c, by2)
    
        Y[:, i+1] = (1 - lm)*Y[:, i] + lm*Y2 + h*gm*(np.matmul(A, Y2) + bvector(x[i] + h*nu + 2*h*mu))
        
    return x, Y
        

def error3(A, bvector, y0, interval, krange):
    
    """ 
    Evaluate the error in the solution for a range of step sizes dictated
    by N = 40k where k = 1:10 in step sizes of 1.
    
    Parameters
    
    ----------
    
    A:  array
        The matrix A as defined in equation (1)
    
    bvector: function
             A function in which the input is the location x and the 
             output is the vector b in equation (1)
    
    y0: array
        n-vector of initial data for the system
    
    interval: list
              A list interval which provides the start and end values
              on which the solution is computed [x0, xn]
              
    krange: list
            A list of the values of k which will be used to evaluate the functions
            
    Returns
    
    -------
    
    Outputs two plots, one for the error vs step-size of the RK3 method, and 
    a second error vs step-size plot for the DIRK3 method. Both plots will be
    log-log, and display the convergence.
    
    """
        
    #Preallocation of error values for each step size:
    e_yrk3 = np.zeros(len(krange))
    e_ydrk3 = np.zeros(len(krange))
        
    h = np.zeros(len(krange))
    
    #Loop through k = 1-10 and call RK3/DIRK3, then compute the error for 
    #each case against the resulting step-size
        
    for i in range(0, len(krange)):
        
        N = 40 * krange[i]
        x, yrk3 = rk3(A, bvector, y0, interval, N)
        x, ydrk3 = dirk3(A, bvector, y0, interval, N)
            
        Y2Re = (1000/999) * (np.exp(-1*x) - np.exp(-1000*x))

            
        h[i] = (interval[1] - interval[0])/N

        
        for j in range(1, int(N)):
            
            e_yrk3[i] = e_yrk3[i] + (h[i] * abs((yrk3[1,j] - Y2Re[j])/Y2Re[j]))
            e_ydrk3[i] = e_ydrk3[i] + (h[i] * abs((ydrk3[1,j] - Y2Re[j])/Y2Re[j]))

    
    
    #Supress first error value from RK3 as it is an outlier
    e_yrk3 = e_yrk3[1:]
    hrk3 = h[1:]

    #Compute the linear trendline
    trend_rk3 = np.polyfit(np.log(hrk3), np.log(e_yrk3), 1)
    trend_drk3 = np.polyfit(np.log(h), np.log(e_ydrk3), 1)

    #Plot both RK3 and DIRK3 on separate plots, as well as the trendline
    plt.figure()
    plt.loglog(hrk3, e_yrk3, 'kx')
    plt.loglog(hrk3, np.exp(trend_rk3[1])*hrk3**(trend_rk3[0]), 'b', label = "Line slope {}".format(np.round(trend_rk3[0],decimals=3)))
    plt.legend()
    plt.xlabel('Step size (h)')
    plt.ylabel('Error')
    plt.title('RK3 Error Q3')


    plt.figure()
    plt.loglog(h, e_ydrk3, 'kx')
    plt.loglog(h, np.exp(trend_drk3[1])*h**(trend_drk3[0]), 'b', label = "Line slope {}".format(np.round(trend_drk3[0], decimals=3)))
    plt.xlabel('Step size (h)')
    plt.ylabel('Error')
    plt.title('DIRK3 Error Q3')
    plt.legend()
    plt.show
    
    
def error4(A, bvector, y0, interval, krange):
    
    """ 
    Evaluate the error in the solution for a range of step sizes dictated
    by N = 200k where k = 4:16 in steps of 1.
        
    
    Parameters
    
    ----------
    
    A:  array
        The matrix A as defined in equation (1)
    
    bvector: function
             A function in which the input is the location x and the 
             output is the vector b in equation (1)
    
    y0: array
        n-vector of initial data for the system
    
    interval: list
              A list interval which provides the start and end values
              on which the solution is computed [x0, xn]
              
    krange: list
            A list of the values of k which will be used to evaluate the functions
            
    Returns
    
    -------
    
    Outputs a plotfor the error vs step-size of the DIRK3 method on a log-log 
    plot along with the convergence of the algorithm.
    
    
    """
        
    #Preallocation of error values for each step size:
    e_ydrk3 = np.zeros((len(krange)))
    h = np.zeros(len(krange))
    
    #Loop through k = 4-16 and call DRK3, then compute the error for 
    #each case against the resulting step-size
        
    for i in range(0, len(krange)):
        
        N = 200 * krange[i]
        x, ydrk3 = dirk3(A, bvector, y0, interval, N)
            
        Y3Re = np.sin(10*x) + 2*np.exp(-1*x) - np.exp(-100*x) - np.exp(-10000*x)

        h[i] = (interval[1] - interval[0])/N
        
        
        for j in range(1, int(N)):
            
            e_ydrk3[i] = e_ydrk3[i] + (h[i] * abs((ydrk3[2,j] - Y3Re[j])/Y3Re[j]))

    #Suppress the first value of error as it is an outlier
    h = h[2:]
    e_ydrk3 = e_ydrk3[2:]
    
    #Compute the trendline of the error vs log
    trend_drk3 = np.polyfit(np.log(h), np.log(e_ydrk3), 1)

    #Plot the error vs step-size, along with the trendline on a log-log plot:
    plt.figure()
    plt.loglog(h, e_ydrk3, 'kx')
    plt.loglog(h, np.exp(trend_drk3[1])*h**(trend_drk3[0]), 'b', label = "Line slope {}".format(np.round(trend_drk3[0], decimals=3)))
    plt.xlabel('Step size (h)')
    plt.ylabel('Error')
    plt.title('DIRK3 Error Q4')
    plt.legend()
    plt.show



def q3():

    """
    
    Function to generate outputs for Q3. Computes the solution using RK3 and
    DRK3, plotting the case where N = 400. Also computes the error in each 
    case and creates two plots containing the eror vs step-size h.
    
    """
    
    def bvector3(x):
        return 0.0

    A = np.array([[-1000, 0],[1000, -1]])
    y0 = np.array([1, 0])
    interval = [0, 0.1]
    
    #Define N just to output a plot for 400 points
    N = 40*10
    
    #Call RK3, and evaluate the exact solution:
    x, Y = rk3(A, bvector3, y0, interval, N)
    
    Y1R = np.exp(-1000*x)
    Y2R = 1000/999 * (np.exp(-1*x) - np.exp(-1000*x))
    
    #RK3 plots:
    
    plt.figure()
    plt.suptitle('RK3 Solution Q3')
    plt.subplot(1,2,1)
    plt.semilogy(x, Y1R, 'rx', label = 'Exact', markersize=3)
    plt.semilogy(x,Y[0,:],'k', label = 'RK3', linewidth = 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_1$')
    plt.ylim([10**-50, 10**0])
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(x, Y2R, 'rx', label = 'Exact',markersize=3)
    plt.plot(x,Y[1,:],'k', label= 'RK3', linewidth = 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_2$')
    plt.legend()
    plt.show
    
    ##########################################################################
    #
    # DIRK3 TESTING:
    #
    ##########################################################################
    
    #Call DIRK3:
    
    x, Y = dirk3(A, bvector3, y0, interval, N)
    
    #DIRK3 plotting:
    
    plt.figure()
    plt.suptitle('DIRK3 Solution Q3')

    plt.subplot(1,2,1)
    plt.semilogy(x,Y[0,:],'rx', label = 'DIRK3', markersize=3)
    plt.semilogy(x, Y1R, 'k', label = 'Real', linewidth=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_1$')
    plt.ylim([10**-50, 10**0])
    plt.legend()
    plt.show
    
    plt.subplot(1,2,2)
    plt.plot(x,Y[1,:],'rx', label = 'DIRK3', markersize=3)
    plt.plot(x, Y2R, 'k', label = 'Real', linewidth=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_2$')
    plt.legend()
    plt.show

    #Call the error function to evaluate the error for both RK3 and DIRK3:    
    error3(A, bvector3, y0, interval, range(1,11))
    

    
def q4():
    
    """
    Solution to the system described in Q4. Same plots as Q3 above however only
    evaluating the error for the DIRK3 algorithm
    """
    
    def bvector4(x):
    
        """
        Defines the b vector for the system defined in Q4
        
        Parameters
        ----------
        x: float
            Value of x to evaluate the B vector
            
        Returns
        -------
        
        Array of size 3 containing the B-vector elements for the system.
        
        """
        
        
        b1 = np.cos(10*x) - (10*np.sin(10*x))
        b2 = 199*np.cos(10*x) - (10*np.sin(10*x))
        b3 = 208*np.cos(10*x) + 10000*np.sin(10*x)
    
        return np.array([b1, b2, b3])
    
    #Define A matrix, initial conditoins, and the solution interval for
    #the 3x3 problem:
    
    A = np.array([[-1, 0, 0], [-99, -100, 0], [-10098, 9900, -10000]])
    y0 = np.array([0, 1, 0])
    interval = [0, 1]
    
    #Define an N = 200*16 for the final plot output:
    N = 200*16
    
    #Call RK3
    x, Y = rk3(A, bvector4, y0, interval, N)
    
    
    # DETERMINE THE REAL SOLUTION:
    Y1R = np.cos(10*x) - np.exp(-x)
    Y2R = np.cos(10*x) + np.exp(-x) - np.exp(-100*x)
    Y3R = np.sin(10*x) + 2*np.exp(-x) - np.exp(-100*x) - np.exp(-10000*x)
    
    #RK3 plots
    plt.figure()
    plt.suptitle('RK3 Solution Q4')

    plt.subplot(1,3,1)
    plt.plot(x, Y1R, 'rx', label = 'Exact', markersize=3)
    plt.plot(x, Y[0,:], 'k', label = 'RK3', linewidth = 1)
    plt.xlabel('x')
    plt.ylabel('y1')
    plt.legend()
    
    plt.subplot(1,3,2)
    plt.plot(x, Y2R, 'rx', label = 'Exact', markersize=3)
    plt.plot(x, Y[1,:], 'k', label = 'RK3', linewidth = 1)
    plt.xlabel('x')
    plt.ylabel('y2')
    plt.legend()
    
    plt.subplot(1,3,3)
    plt.plot(x, Y3R, 'rx', label = 'Exact', markersize=3)
    plt.plot(x, Y[2,:], 'k', label = 'RK3', linewidth = 1)
    plt.xlabel('x')
    plt.ylabel('y3')
    plt.legend()

    ##########################################################################
    #
    # DIRK3 SOLUTION:    
    #
    ##########################################################################
    
    #Call DIRK3
    x, Ydirk3 = dirk3(A, bvector4, y0, interval, N)
    
    #Evaluate DIRK3 error:
    error4(A, bvector4, y0, interval, range(4, 17))
    
    #DIRK3 plots:
    
    plt.figure()
    plt.suptitle('DIRK3 Solution Q4')

    plt.subplot(1,3,1)
    plt.plot(x, Y1R, 'rx', label = 'Exact', markersize=3)
    plt.plot(x, Ydirk3[0,:], 'k', label = 'DIRK3', linewidth = 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_1$')
    plt.legend()

    plt.subplot(1,3,2)
    plt.plot(x, Y2R, 'rx', label = 'Exact', markersize=3)
    plt.plot(x, Ydirk3[1,:], 'k', label = 'DIRK3', linewidth = 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_2$')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(x, Y3R, 'rx', label = 'Exact', markersize=3)
    plt.plot(x, Ydirk3[2,:], 'k', label = 'DIRK3', linewidth = 1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y_3$')
    plt.legend()


##############################################################################
#
# EXECUTE SCRIPT WHEN RUN 
#
##############################################################################

if __name__ == "__main__":
    
    q3()
    q4()