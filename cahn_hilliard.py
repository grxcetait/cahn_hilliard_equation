#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:31:06 2026

@author: gracetait
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import os

class CahnHilliard(object):

    def __init__(self, a, phi, l, M, dx, dt, kappa):
        
        # Define parameters
        self.a = a
        #self.n = n
        self.l = l
        self.M = M
        self.dx = dx
        self.dt = dt
        self.kappa = kappa
        self.phi_value = phi
        self.phi = self.init_phi()
        #self.mu = np.zeros((self.l, self.l))
        self.calculate_mu()
        
        
    def init_phi(self):
        
        #noise = np.random.uniform(self.phi_value - 0.1, self.phi_value + 0.1, size = (self.l, self.l))

        #return np.ones((self.l, self.l)) * self.phi_value + noise
        return np.random.uniform(self.phi_value - 0.1, self.phi_value + 0.1, size = (self.l, self.l))
        
    def laplacian(self, array):
        
        laplacian = np.roll(array, 1, axis = 0) + np.roll(array, 1, axis = 1) + \
            np.roll(array, -1, axis = 0) + np.roll(array, -1, axis = 1) - 4 * array
            
        return laplacian
        
    def calculate_mu(self):
        
        self.mu = - self.phi * (1 - self.phi**2) - self.laplacian(self.phi)
            
    def calculate_phi(self):
        
        self.calculate_mu()
        
        self.phi = self.phi + self.M * self.dt * self.laplacian(self.mu) / self.dx**2
        
    def calculate_free_energy_density(self):
        
        self.free_energy_density = - self.a * self.phi**2 / 2 + \
            self.a ** self.phi**4 / 4 + self.kappa * self.laplacian(self.phi)**2 / 2
            
    def animate(self, steps):
        
        # Define the figure and axes for the animaΩtion
        fig, ax = plt.subplots()
        
        # Initialise the image object
        im = ax.imshow(self.phi, cmap = "magma",
                       vmin = -1, vmax = 1)
        plt.colorbar(im)
        
        # Run the animation for the total number of steps
        for s in range(steps):
            
            # Update the array
            self.calculate_phi()
            
            if s % 100 == 0:
            
                # Update the animation
                im.set_data(self.phi)
                ax.set_title(f"Step: {s}")
            
                # Keep the image up while the script is running
                plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
            
cahn_hilliard = CahnHilliard(a = 0.1, phi = 0.5, l = 50, M = 0.1, 
                             dx = 1, dt = 0.1, kappa = 0.1)
init = cahn_hilliard.animate(steps = 100000)