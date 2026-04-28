#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:31:06 2026

@author: gracetait
"""

import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
from numba import njit

@njit
def laplacian_numba(array, n):
    """
    Compute the discrete unnormalised Laplacian of a 2D array with periodic 
    boundary conditions.

    Parameters
    ----------
    array : np.ndarray of shape (n x n)
        Input scalar field.
    n : int
        Length of lattice of shape (n x n)

    Returns
    -------
    np.ndarray of shape (n x n)
        Discrete Laplacian (unnormalised by dx²).
    """

    # Make an empty copy of the lattice 
    output = np.empty_like(array)
    
    # Iterate through lattice
    for i in range(n):
        for j in range(n):
            
            # Explicit periodic boundary indexing
            up = array[(i - 1) % n, j]
            down = array[(i + 1) % n, j]
            left = array[i, (j - 1) % n]
            right = array[i, (j + 1) % n]
            output[i, j] = up + down + left + right - 4 * array[i, j]
            
    return output

@njit
def ch_step_numba(phi, n, dx, dt, a, k, M):
    """
    Performs one full Cahn-Hilliard time step in compiled code.
    
    Parameters
    ----------
    phi : np.ndarray of shape (n x n)
        The composition field
    n : int
        Length of lattice of shape (n x n)
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    a : float
        Constant 'a'.
    k : float
        Constant 'k'.
    M : float
        Constant 'M'.

    Returns
    -------
    np.ndarray of shape (n x n)
        Updated composition field.
    """

    # Calculate the chemical potenital field 
    # mu = -a * phi + a * phi^3 - k * laplacian(phi) / dx^2
    lap_phi = laplacian_numba(phi, n)
    mu = - a * phi + a * phi**3 - k * lap_phi / dx**2
    
    # Update phi by one time step
    # phi_new = phi + M * dt * laplacian(mu) / dx^2
    lap_mu = laplacian_numba(mu, n)
    phi_new = phi + (M * dt * lap_mu) / dx**2
    
    return phi_new

@njit
def run_ch_block_numba(phi, n, dx, dt, a, k, M, block_size):
    """
    Runs a block of steps entirely in Numba.
    
    Parameters
    ----------
    phi : np.ndarray of shape (n x n)
        The composition field
    n : int
        Length of lattice of shape (n x n)
    dx : float
        Spatial step size.
    dt : float
        Time step size.
    a : float
        Constant 'a'.
    k : float
        Constant 'k'.
    M : float
        Constant 'M'.
    block_size : int
        Number of simulation steps.

    Returns
    -------
    np.ndarray of shape (n x n)
        Updated composition field.
    """
    
    # Save current phi
    current_phi = phi
    
    # Iterate through a block of steps
    for _ in range(block_size):
        
        # Run one full time step update
        current_phi = ch_step_numba(current_phi, n, dx, dt, a, k, M)
        
    return current_phi

class CahnHilliard(object):
    """
    A class that implements the Cahn-Hilliard equation for phase separation in
    a physical system (eg. oil-water mixtures).
    """

    def __init__(self, phi, l, dx, dt, a, k, M):
        """
        Initialise the Cahn-Hilliard system

        Parameters
        ----------
        phi : float
            Mean initial composition. The field is seeded with uniform
            random noise in the range [phi - 0.1, phi + 0.1].
        l : int
            Side length of the square lattice.
        dx : float
            Spatial step size.
        dt : float
            Time step size.
        a : float
            Constant 'a'.
        k : float
            Constant 'k'.
        M : float
            Constant 'M'.

        Returns
        -------
        None.

        """
        
        # Define parameters
        self.l = l
        self.dx = dx
        self.dt = dt
        self.phi_value = phi
        self.phi = self.init_phi()
        self.a = a
        self.k = k
        self.M = M
        
    def init_phi(self):
        """
        Initialise the field with small random perturbations where the values
        are drawn from a uniform distribution around the inputted "phi_value"
        with some small random noise.

        Returns
        -------
        np.ndarray of shape (l, l)
            Randomly initialised composition field.

        """
        
        return np.random.uniform(self.phi_value - 0.1, self.phi_value + 0.1, 
                                 size = (self.l, self.l)).astype(np.float64)
            
    def calculate_phi(self):
        """
        Advance the composition field using Numba.

        Returns
        -------
        None.

        """
        
        self.phi = ch_step_numba(self.phi, self.l, self.dx, self.dt, self.a, self.k, self.M)
        
    def calculate_free_energy_density(self):
        """
        Compute the free energy density across the lattice.

        Returns
        -------
        np.ndarray of shape (l, l)
            Free energy density at each grid point.

        """
        
        grad_x = (np.roll(self.phi, -1, axis=0) - np.roll(self.phi, 1, axis=0)) / (2 * self.dx)
        grad_y = (np.roll(self.phi, -1, axis=1) - np.roll(self.phi, 1, axis=1)) / (2 * self.dx)
        grad_sq = grad_x**2 + grad_y**2
   
        return - self.a * self.phi**2 / 2 + self.a * self.phi**4 / 4 + self.k * grad_sq / 2
        
    
class Simulation(object):
    """
    A class to handle the execution, measurement, and visualisation 
    of the Cahn-Hilliard simulation.
    """
    
    def __init__(self, phi, l, dx, dt, a, k, M, steps, mea_int):
        """
        Initialise simulation parameters

        Parameters
        ----------
        phi : float
            Mean initial composition (order parameter).
        l : int
            Side length of the square lattice (l x l grid points).
        dx : float
            Spatial step size.
        dt : float
            Time step size.
        a : float
            Constant 'a'.
        k : float
            Constant 'k'.
        M : float
            Constant 'M'.
        steps : int
            Number of measurement steps or animation frames.
        mea_int : int
            Interval to take measurements.

        Returns
        -------
        None.

        """
        
        # Define parameters
        self.l = l
        self.dx = dx
        self.dt = dt
        self.phi = phi
        self.a = a
        self.k = k
        self.M = M
        self.steps = steps
        self.mea_int = mea_int
        
    def animate(self):
        """
        Run and display an animation of the Cahn-Hilliard simulation.

        Parameters
        ----------
        steps : int
            Total number of time steps to simulate.

        Returns
        -------
        None.

        """
        
        # Initialise the lattice using the CahnHilliard class
        ch = CahnHilliard(self.phi, self.l, self.dx, self.dt, self.a, self.k, self.M)
        
        # Define the figure and axes for the animaΩtion
        fig, ax = plt.subplots()
        
        # Initialise the image object
        im = ax.imshow(ch.phi, cmap = "magma",
                       vmin = -1, vmax = 1)
        plt.colorbar(im)
        
        # Run the animation for the total number of steps
        for s in range(self.steps):
            
            # Update the array
            ch.calculate_phi()
            
            # Update the animation every 100 steps
            if s % 100 == 0:
            
                # Update the animation
                im.set_data(ch.phi)
                ax.set_title(f"Step: {s}")
            
                # Keep the image up while the script is running
                plt.pause(0.001)
            
        # Keep the final image open when the loop finishes
        plt.show()
        
    def measurements(self, filename):
        """
        Run the simulation and record the mean free energy density over time.

        Parameters
        ----------
        filename : str
            Name of the output data file (e.g. ``"ch_data.txt"``).
        steps : int
            Total number of time steps to simulate.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path = os.path.join(datafiles_folder, filename)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Make empty list to hold data points
        free_energy_density = []
        time = []
        
        # Initialise the lattice using the CahnHilliard class
        ch = CahnHilliard(self.phi, self.l, self.dx, self.dt, self.a, self.k, self.M)
        
        # Iterate through simulation steps
        for s in range(self.steps // self.mea_int):
            step = s * self.mea_int
            print(f"\rSimulating step = {step}/{self.steps}", end='', flush=True)
            
            # Run mea_init steps at once in Numba before returning to Python to measure
            ch.phi = run_ch_block_numba(ch.phi, ch.l, ch.dx, ch.dt, ch.a, ch.k, ch.M, self.mea_int)
            
            fed = np.mean(ch.calculate_free_energy_density())
            free_energy_density.append(fed)
            time.append(step)
            
        print()
            
        # Open in "a" (append) or "w" (overwrite) mode
        # Write the values into the specified file
        with open(file_path, "w") as f:
            for i in range(len(time)):
                
                f.write(f"{free_energy_density[i]},{time[i]}\n")
                
    def plot_measurements(self, filename):
        """
        Generate and save a plot of the mean free energy density over time data.

        Parameters
        ----------
        filename : str
            The data file to read from.

        Returns
        -------
        None.

        """
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        filename_path = os.path.join(outputs_directory, "datafiles", filename)
        plots_folder = os.path.join(outputs_directory, "plots")
        
        # If the folders dont exist, create them
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # Create an empty list to store input data
        input_data = []        

        # Read in the data from the specified text file
        try:
            with open(filename_path, "r") as filein:
                for line in filein:
                    input_data.extend(line.strip(" \n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            
        # Make empty list to store data
        free_energy_density = []
        time = []
        
        # Iterate through input data and append to empty lists
        for i in range(0, len(input_data), 2):
            
            # Obtain vlaue from input data
            fed = float(input_data[i])
            t = float(input_data[i+1])
            
            # Append to lists
            free_energy_density.append(fed)
            time.append(t)
        
        # Create empty plots
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 10))
        
        ax1.plot(time, free_energy_density)
        ax1.set_ylabel(r"Mean Free Energy Density $\langle f(\phi) \rangle$", fontsize = 14)
        ax1.set_xlabel(r"Time $t = dt$", fontsize = 14)
        ax1.set_title(
    rf"Free energy density vs time with $\phi$ = {self.phi}, dx = {self.dx}, dt = {self.dt}"
    "\n" 
    rf"for a {self.l} x {self.l} lattice", 
    fontsize=16
)
        #ax1.set_suptitle(f"$\phi$ = {self.phi}")
        
        # Fix any overlapping labels, titles or tick marks
        plt.tight_layout()
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plot successfully saved to: {save_path}")
        
        # Show final plots
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Cahn Hilliard")
    
    # User input parameters
    parser.add_argument("--phi", type = float, default = 0, help = "Initial oil to water ratio")
    parser.add_argument("--l", type = int, default = 100, help = "Lattice size (l x l)")
    parser.add_argument("--dx", type = float, default = 1, help = "Spatial step")
    parser.add_argument("--dt", type = float, default = 0.01, help = "Time step")
    parser.add_argument("--a", type = float, default = 1, help = "Constant 'a'")
    parser.add_argument("--k", type = float, default = 1, help = "Constant 'k'")
    parser.add_argument("--M", type = float, default = 1, help = "Constant 'M'")
    parser.add_argument("--mode", type = str, default = "ani", choices = ["ani", "mea"],
                         help = "Animation or measurements")
    parser.add_argument("--steps", type = int, default = 20000,
                        help = "Number of measurement steps or animation frames.")
    parser.add_argument("--int", type = int, default = 100,
                        help = "Interval to take measurements.")
    
    args = parser.parse_args()
    
    # Pass in parameters to the Simulation class
    sim = Simulation(phi = args.phi, l = args.l, dx = args.dx, dt = args.dt, 
                     a = args.a, k = args.k, M = args.M, steps = args.steps,
                     mea_int = args.int)
        
    if args.mode == "ani":
    
        sim.animate()
        
    else:
        
        filename = f"ch_free_energy_density_{args.steps}steps_{args.phi}phi_{args.dx}dx_{args.dt}dt.txt"
        sim.measurements(filename)
        sim.plot_measurements(filename)
    
