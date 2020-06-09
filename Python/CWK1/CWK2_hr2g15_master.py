#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:40:17 2019

@author: haider

MATH6141 COURSEWORK 2

Algorithm chosen as the black-box scipy.solve_bvp routine. This is primarily
due to the fact that as it is an available package it has been well-tested and
requires less development time (for a final year student drowning in work +
job applications). The algorithm utilises a collocation algorithm which therefore
provides faster convergence compared to other methods such as finite differencing.

The system is also non-linear and therefore eliminates the shooting method

The drawback is some awkward definitions within the scipy routine such as 
the specifying of residuals of the system, and having to strangely define
some initial guesses to ensure the correct solution is captured.

"""

import numpy as np
from scipy.integrate import solve_bvp, odeint
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def hair_sys(s, STATE, fg, fx):
    
    """
    This function will linerise the system of 2 second-order ODEs into a system
    of 4 first order ODEs in order to solve the boundary value problem
    
    Parameters
    ----------
    s:  float
        Inpendent vairable describing arc-length along the hair
        
    STATE:  Array
            An array of length 4 with the initial conditions of each element 
            1st value: theta 
            2nd value: theta dot
            3rd value: phi
            4th value: phi dot
            
    fg: float
        Gravity term in equation
        
    fx: float
        Wind term in equation
        
    
    Returns
    -------
    
    State_out:  Array
                An array of the derivates of the state variables which will
                be integrated.
    """


    #Paritiion the 'state vector' to the components needed in the equation:
    
    th1 = STATE[0]
    th2 = STATE[1]
    ph1 = STATE[2]
    ph2 = STATE[3]
    
    th1d = th2
    ph1d = ph2
    
    th2d = s*fg*np.cos(th1) + s*fx*np.cos(ph1)*np.sin(th1)
    ph2d= -s*fx*np.sin(ph1)*np.sin(th1)
    
    return np.array([th1d, th2d, ph1d, ph2d])



def hair_sys_theta(s, STATE, fg, fx):
    """
    This function will linerise the system of 1 second-order ODE of theta into a system
    of 2 first order ODEs in order to solve the boundary value problem
    
    Parameters
    ----------
    s:  float
        Inpendent vairable describing arc-length along the hair
        
    STATE:  Array
            An array of length 2 with the initial conditions of each element 
            
            1st value: theta 
            2nd value: theta dot
            
    fg: float
        Gravity term in equation
        
    fx: float
        Wind term in equation
        
    
    Returns
    -------
    
    State_out:  Array
                An array of the derivates of the state variables which will
                be integrated.
    """

    
    th1 = STATE[0]
    th2 = STATE[1]
    
    th1d = th2
    th2d = s * fg * np.cos(th1) + s*fx*np.sin(th1)
    
    return np.array([th1d, th2d])


def hair_cart(R, th0, phi0):
    
    """
    Function to define the initial position of hair in the cartesian frame, given 
    values of the radius, and the longitudinal angle theta and lateral angle
    phi.
    
    Parameters
    ----------
    
    R:  float
        Radius of the heat
    
    th0: float
         Longitudinal angle theta 
    
    ph0: float
         Lateral angle phi
         
    Returns
    -------
    
    out: array
         An array containing the initial positions in the cartesian frame.
    
    
    """
    
    return np.array([R*np.cos(th0)*np.cos(phi0), -R*np.cos(th0)*np.sin(phi0), R*np.sin(th0)])


def cart_xyz(s, STATE, res_theta):
    
    """
    Defines the 3DOF Cartesian system for hair in the xyz plane.
    
    Parameters
    ----------
    
    s:  float
        Independent variable describing arc-length along hair
    
    STATE: array
           State vector containing positions 
    
    res_theta:  solve_bvp object
                The solve_bvp handle passed in order to call the res_theta.sol
                routine to interpolate the solution of theta(s)/phi(s)
         
    Returns
    -------
    
    out: array
         An array containing the derivatives of positions in the cartesian frame.
    
    
    """
    
    dxds = np.cos(res_theta.sol(s)[0]) * np.cos(res_theta.sol(s)[2])
    dyds = np.cos(res_theta.sol(s)[0]) * np.sin(res_theta.sol(s)[2])
    dzds = np.sin(res_theta.sol(s)[0])
    
    return np.array([dxds, dyds, dzds])



def ic_cartxz(R, th0):
    
    """
    Function to define the initial position of hair in the cartesian frame, given 
    values of the radius, and the longitudinal angle theta. This is for the 
    xz system only where phi is 0.
    
    Parameters
    ----------
    
    R:  float
        Radius of the head
    
    th0: float
         Longitudinal angle theta 

         
    Returns
    -------
    
    out: array
         An array containing the initial positions in the cartesian frame.
    
    
    """
    
    return np.array([R*np.cos(th0), R*np.sin(th0)])


def cart_xz(STATE, s, res_theta):
    
    """
    Defines the 2DOF Cartesian system for hair in the xz plane.
    
    Parameters
    ----------
    
    s:  float
        Independent variable describing arc-length along hair
    
    STATE: array
           State vector containing positions 
    
    res_theta:  solve_bvp object
                The solve_bvp handle passed in order to call the res_theta.sol
                routine to interpolate the solution of theta(s)
         
    Returns
    -------
    
    out: array
         An array containing the derivatives of positions in the cartesian frame.
    
    
    """
    
    dxds = np.cos(res_theta.sol(s)[0])
    dzds = np.sin(res_theta.sol(s)[0])
    
    return np.array([dxds, dzds])



def hair_xz_main(L, R, fg, fx, theta):
    
    """
    Top level function which, given values of hair length, head radius, wind speed,
    gravity, and theta values, computes the locations of hairs in the cartesian
    frame by solving first the BVP using scipy.solve_bvp then the IVP using 
    scipy.integrate.odeint. The interpolated solve_bvp polynomial solution 
    is passed to the IVP function in order to evaluate theta. 
    
    [This is the solution to Q1]
    
    Parameters
    ----------
    
    L:  float
        Length of the hairs. This will bound the BVP solution domain in 
        [0, L]
    
    R:  float
        Radius of the head. This will specify the initial conditions in the 
        cartesian frame of reference.
        
    fg: float
        Constant describing the gravity term in the hair-modelling equation
        
    fx: float
        Consant describing the wind constant term in the hair-modelling equation
        
    theta:  array
            An array of values of the angle theta which is defined as the latitude
    
    Returns
    -------
    
    x_hair_ivp:    array
                   An array of the x coordinates of hair, dimension (NxM) where 
                   N is the number of spatial steps used, and M is the number of hairs
                   in the model.

    z_hair_ivp:    array
                   An array of the z coordinates of hair, dimension (NxM) where 
                   N is the number of spatial steps used, and M is the number of hairs
                   in the model.                   
    
    
    Calling this function will execute several sub-functions which compute and
    return a plot of the hairs in the x-z frame. 
    
    
    """
    ###################################
    #
    # ASSERTS GO HERE TO CHECK FUNCTION:
    #
    ####################################
    
    #Check L is non-zero and positive
    
    assert(L!=0), "L must be a non-zero value"
    assert(L > 0), "L must be greater than 0"
    
    #Check R is non-zero and positive
    
    assert(R!=0), "R must be a non-zero value"
    assert(R > 0), "R must be greater than 0"

    
    #Check fg is a number
    
    assert(isinstance(fg, float)), "fg must be a defined integer or float"
    assert(isinstance(fx, float)), "fx must be a defined integer or float"

    #Check if theta is an array
    
    assert(isinstance(theta, np.ndarray)), "theta must be an array"
    assert(theta[0] != theta[-1]), "theta must be a defined range"
    
    
    print('Initialising solution-space...')
    
    bcstheta = theta
    s_init = np.linspace(0, L, 20000)
    y_init = np.zeros((2, s_init.size))
    
    y_sol_bvp = np.zeros((s_init.size, bcstheta.size))
    x_sol_bvp = np.zeros((s_init.size, bcstheta.size))
    x_sol_ivp = np.zeros((s_init.size, bcstheta.size))
    z_sol_ivp = np.zeros((s_init.size, bcstheta.size))
    
    
    def bc_theta2(ya, yb):
        
        #Return the residual of the boundary conditions, thus subtract the value:
        #theta = p[0]
        
        a = np.array([(ya[0] - bcsthet), yb[1]])
        
        return a
    
    #Main program loop, for each initial value solve BVP and IVP
    for i in range(0, len(bcstheta)):
        
        print('Solving for hair: ', i+1)
        #Index the relevant intiial theta position on the head
        bcsthet = bcstheta[i]
        
        #Call solve_bvp, store solution array
       # res_theta = solve_bvp(hair_sys_theta3, bc_theta2, s_init, y_init, args = (fg,fx))
       
        res_theta = solve_bvp((lambda s, ST: hair_sys_theta(s, ST, fg, fx)), bc_theta2, s_init, y_init, max_nodes=20000)
        #print(res_theta.message)
       
        #print(res_theta.message)
        x_sol_bvp[:, i] = res_theta.x
        y_sol_bvp[:, i] = res_theta.y[0, :]
        
        
        #Update the initial guess to be the value of theta being solved        
        if i > (len(bcstheta)-3)/2:
            y_init[0, :] = bcsthet + 0.5*np.pi
        
        #Set the cartesian initial condition for this case:
        cart_ic = ic_cartxz(R, bcsthet)

        #Call ODEINT (FORTRAN MASTER RACE) to solve the ivp, store solution:
        STATE = odeint(cart_xz, cart_ic, s_init, args = (res_theta,))
        
        x_sol_ivp[:, i] = STATE[:, 0]
        z_sol_ivp[:, i] = STATE[:, 1]
        
        #return x_sol_ivp, z_sol_ivp
    
    #Plot outputs:
    
    plt.figure()
    th = np.linspace(0, 2*np.pi,1000)
    xx = 10*np.cos(th)
    yy = 10*np.sin(th)
    plt.plot(xx,yy,'k')
    #plt.xlim([-15, 15])
    #plt.ylim([-15, 15])
    plt.axis('Equal')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$z$')
    
    if fx != 0:    
        plt.title('Q3: Wind')
    elif fx == 0:
        plt.title('Q2: No wind')   
    
    for i in range(0, x_sol_ivp.shape[1]):
        plt.plot(x_sol_ivp[:, i], z_sol_ivp[:, i], 'b')

    #plt.figure()
    #for i in range(0, x_sol_ivp.shape[1]):
    #    plt.plot(x_sol_bvp[:,i], y_sol_bvp[:,i])

    print('Complete')
    return x_sol_ivp, z_sol_ivp



def hair_xyz_main(L, R, fg, fx, theta, phi):
    


    """
    Top level function which, given values of hair length, head radius, wind speed,
    gravity, and theta values, computes the locations of hairs in the cartesian
    frame by solving first the BVP using scipy.solve_bvp then the IVP using 
    scipy.integrate.odeint. The interpolated solve_bvp polynomial solution 
    is passed to the IVP function in order to evaluate theta and phi.
    Each hair position will be looped through and its
    values stored in pre-allocated numpy arrays. 
    
    [This is the solution to Q4]
    
    Parameters
    ----------
    
    L:  float
        Length of the hairs. This will bound the BVP/IVP solution domain in 
        [0, L]
    
    R:  float
        Radius of the head. This will specify the initial conditions in the 
        cartesian frame of reference.
        
    fg: float
        Constant describing the gravity term in the hair-modelling equation
        
    fx: float
        Consant describing the wind constant term in the hair-modelling equation
        
    theta:  array
            An array of values of the angle theta which is defined as the latitude
            
    phi:    array
            An array of values of the angle phi which is defined as the longitude
    
    Returns
    -------
    
    x_hair_ivp:    array
                   An array of the x coordinates of hair, dimension (NxM) where 
                   N is the number of spatial steps used, and M is the number of hairs
                   in the model.

    y_hair_ivp:    array
                   An array of the y coordinates of hair, dimension (NxM) where 
                   N is the number of spatial steps used, and M is the number of hairs
                   in the model. 


    z_hair_ivp:    array
                   An array of the z coordinates of hair, dimension (NxM) where 
                   N is the number of spatial steps used, and M is the number of hairs
                   in the model.                   
    
    
    Calling this function will execute several sub-functions which compute and
    return a plot of the hairs in the x-z frame, x-y frame, as well as an interactive
    3D plot displaying all hairs. 
    
    
    """
    
    ###################################
    #
    # ASSERTS GO HERE TO CHECK FUNCTION:
    #
    ####################################
    
    #Check L, R are non-zero and positive
    
    assert(L!=0), "L must be a non-zero value"
    assert(L > 0), "L must be greater than 0"
    assert(R!=0), "R must be a non-zero value"
    assert(R > 0), "R must be greater than 0"

    
    #Check fg/fx is a number
    
    assert(isinstance(fg, float)), "fg must be a defined"
    assert(isinstance(fx, float)), "fx must be a defined integer or float"

    
    #Check if theta/phi is an array
    
    assert(isinstance(theta, np.ndarray)), "theta must be an array"
    assert(theta[0] != theta[-1]), "theta must be a defined range"
    assert(isinstance(phi, np.ndarray)), "phi must be an array"
    
    ##########################################################################
    
    print('Initialising solution-space')
    s_init = np.linspace(0, L, 20000)
    y_init = np.zeros((4, s_init.size))

    #Initialise angular components and construct the matrix of initial conditions:    
    bcsthet = theta
    #bcsphi = np.zeros_like(bcsthet)
    bcsphi = phi
    init = np.zeros((len(bcsthet),2))
    init[:, 0] = bcsthet
    init[:, 1] = bcsphi
    
    #Pre-allocate solution arrays for efficiency:
    #y_sol_bvp = np.zeros((s_init.size, bcstheta.size))
    x_sol_bvp = np.zeros((s_init.size, 100))
    x_sol_ivp = np.zeros((s_init.size, 100))
    y_sol_ivp = np.zeros((s_init.size, 100))
    z_sol_ivp = np.zeros((s_init.size, 100))
    
    
    def bc_system(ya, yb):
        
        """
        Residuals for the full coupled system
        """
        #4 dimension residual

        return np.array([(ya[0] - init_th), yb[1], (ya[2] - init_ph), yb[3]])
    
    #Initialise hair counter:
    k = 0
    
    #Major loop through theta 
    
    for i in range(0, len(init)):

        #Minor loop through phi for each theta 
        
        for j in range(0, len(init)):
            print('Solving for hair: ', k+1)
            
            #Extract the [i,j] initial condition for the BVP for this loop:
            init_th = init[i, 0]
            init_ph = init[j, 1]
        
            #Call Solve BVP
            res_theta = solve_bvp((lambda s, ST: hair_sys(s, ST, fg, fx)), bc_system, s_init, y_init, max_nodes=20000)
            x_sol_bvp[:, k] = res_theta.x
            #print(res_theta.message)
            #Adjust the BVP guess to be the initial value across the mesh
            
            #y_init[0, :] = init_th
            #y_init[2, :] = init_ph
            
            #if i > (len(init)-3)/2:
             #   y_init[0, :] = init_th + 0.5*np.pi
              #  y_init[2, :] = init_ph + 0.5*np.pi

            #Call initial conditions routine in cartesian FOR:
            cart_ic = hair_cart(R, init_th, init_ph)
    
            #Call ODEINT (FORTRAN MASTER RACE), solve for cartesian coordinates
            STATE = odeint(cart_xyz, cart_ic, s_init, args = (res_theta,), tfirst=True)
    
            #Extract states
            x_sol_ivp[:, k] = STATE[:, 0]
            y_sol_ivp[:, k] = STATE[:, 1]
            z_sol_ivp[:, k] = STATE[:, 2]
            
            #Loop to next hair for storing
            k = k + 1
    
    #Plot in xz & xy plane:
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(x_sol_ivp, z_sol_ivp)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.title('Q5 x-z plane plot')
    plt.subplot(1,2,2)
    plt.plot(x_sol_ivp, y_sol_ivp)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-15, 15])
    plt.ylim([-15, 15])
    plt.title('Q5 x-y plane plot')
    plt.tight_layout()
    
    #Plot in 3 dimensions:

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-15, 15)
    ax.set_ylim3d(-15, 15)
    ax.set_zlim3d(-15, 15)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Q5 plot')
    
    for i in range(0, x_sol_ivp.shape[1]):

        ax.plot(x_sol_ivp[:, i], y_sol_ivp[:, i], z_sol_ivp[:, i])

    print('Complete')
    return x_sol_ivp, y_sol_ivp, z_sol_ivp



#Run the script when called
if __name__ == "__main__":
    
    #Run algorithm for Q2:
    xhair2, zhair2 = hair_xz_main(4, 10, 0.1, 0.0, np.linspace(0, np.pi,20))
    
    #Run algorithm for Q3:
    xhair3, zhair3 = hair_xz_main(4, 10, 0.1, 0.1, np.linspace(0, np.pi,20))

    #Run algorithm for Q5:
    xhair5, yhair5, zhair5 = hair_xyz_main(4, 10, 0.1, 0.05, np.linspace(0, 0.49*np.pi, 10), np.linspace(0, np.pi,10))
    
