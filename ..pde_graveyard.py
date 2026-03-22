#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 17:12:13 2026

@author: gracetait
"""

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
        
        # Update the animation every 100 steps
        if s % 100 == 0:
        
            # Update the animation
            im.set_data(self.phi)
            ax.set_title(f"Step: {s}")
        
            # Keep the image up while the script is running
            plt.pause(0.001)
        
    # Keep the final image open when the loop finishes
    plt.show()
    
# Create empty lists to store data
phi_list = []
x_list = []
y_list = []

# Iterate through input data and append to empty lists
for i in range(0, len(input_data), 2):
    
    # Obtain vlaue from input data
    phi = float(input_data[i])
    x = float(input_data[i+1])
    y = float(input_data[i+2])     
    
    # Append to lists
    phi_list.append(phi)
    x_list.append(phi)
    y_list.append(phi)
    
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse


class Poisson(object):

    def __init__(self, l, tolerance, omega):

        # Define parameters
        self.l = l
        self.omega = omega
        self.tolerance = tolerance
        self.phi = self.init_phi()
        self.rho = self.init_rho()

    def boundary_conditions(self, phi):

        # Set boundaries to be zero
        phi[0, :, :] = 0  # x-axis top face
        phi[-1, :, :] = 0  # x-axis bottom face
        phi[:, 0, :] = 0  # y-axis top face
        phi[:, -1, :] = 0  # y-axis bottom face
        phi[:, :, 0] = 0  # z-axis top face
        phi[:, :, -1] = 0  # z-axis bottom face

        return phi

    def init_phi(self):

        # Initialise lattice to have some random noise between 0 and 1
        phi = np.random.rand(self.l, self.l, self.l)
        #phi = np.zeros(shape = (self.l, self.l, self.l))

        # Assign to self.phi and set boundaries to be zero
        return self.boundary_conditions(phi)

    def init_rho(self):

        # Initiate rho as a monopole
        rho = np.zeros(shape=(self.l, self.l, self.l))
        rho[self.l // 2, self.l // 2, self.l // 2] = 1

        return rho

    def jacobi(self):

        # Calculate new phi, taking into consideration periodic boundaries
        new_phi = (np.roll(self.phi, 1, axis=0) + np.roll(self.phi, -1, axis=0) +
                   np.roll(self.phi, 1, axis=1) + np.roll(self.phi, -1, axis=1) +
                   np.roll(self.phi, 1, axis=2) + np.roll(self.phi, -1, axis=2) +
                   self.rho) / 6

        # Set boundaries to be zero
        new_phi = self.boundary_conditions(new_phi)

        # Calculate distance between old and new phi
        distance = np.abs(self.phi - new_phi)

        # Update phi 
        self.phi = new_phi

        return distance

    def gauss_seidel(self):

        # First, save old phi
        old_phi = self.phi.copy()

        # Calculate new phi using the most recently updated cells
        # Avoiding the boundaries so they stay at zero
        for i in range(1, self.l - 1):
            for j in range(1, self.l - 1):
                for k in range(1, self.l - 1):

                    self.phi[i, j, k] = (
                        self.phi[i - 1, j, k] + self.phi[i + 1, j, k] +
                        self.phi[i, j - 1, k] + self.phi[i, j + 1, k] +
                        self.phi[i, j, k - 1] + self.phi[i, j, k + 1] +
                        self.rho[i, j, k]) / 6

        # Calculate distance between old and new phi
        distance = np.abs(self.phi - old_phi)

        return distance

    def sor(self):

        # First, save old phi
        old_phi = self.phi.copy()

        # Calculate new phi using over relaxation
        # Avoiding the boundaries so they stay at zero
        for i in range(1, self.l - 1):
            for j in range(1, self.l - 1):
                for k in range(1, self.l - 1):

                    gs = (self.phi[i - 1, j, k] + self.phi[i + 1, j, k] +
                            self.phi[i, j - 1, k] + self.phi[i, j + 1, k] +
                            self.phi[i, j, k - 1] + self.phi[i, j, k + 1] +
                            self.rho[i, j, k]) / 6
                    
                    self.phi[i, j, k] = (1 - self.omega) * self.phi[i, j, k] +\
                        self.omega * gs

        # Calculate distance between old and new phi
        distance = np.abs(self.phi - old_phi)

        return distance

    def get_electric_field(self):
        
        grad_x, grad_y, grad_z = np.gradient(self.phi)

        return -grad_x, -grad_y, -grad_z

class Simulation(object):
    
    def __init__(self, l, tolerance, omega):

        # Define parameters
        self.l = l
        self.omega = omega
        self.tolerance = tolerance
            
    def measurements(self, alg, filename1, filename2, filename3, filename4):
        
        # Define datafiles output directory
        base_directory = os.path.dirname(os.path.abspath(__file__))
        outputs_directory = os.path.join(base_directory, "outputs")
        datafiles_folder = os.path.join(outputs_directory, "datafiles")
        file_path1 = os.path.join(datafiles_folder, filename1)
        file_path2 = os.path.join(datafiles_folder, filename2)
        file_path3 = os.path.join(datafiles_folder, filename3)
        file_path4 = os.path.join(datafiles_folder, filename4)
        
        # If the folders don't exist, create them
        if not os.path.exists(datafiles_folder):
            os.makedirs(datafiles_folder)
            
        # Initialise Poisson class
        poisson = Poisson(self.l, self.tolerance, self.omega)

        # Define algorithm
        if alg == "j":
            update = poisson.jacobi

        elif alg == "gs":
            update = poisson.gauss_seidel

        else:
            update = poisson.sor

        # Update phi and obtain the distance between old and new phi
        distance = update()

        # Let the error be the largest distance
        error = np.max(distance)

        # Set time to zero
        t = 0

        # Continue to update phi until the error is smaller than the tolerance
        while error > self.tolerance:
            print(f"Simulating step = {t}", end = '\r')

            # Update the animation and obtain the error
            distance = update()
            error = np.max(distance)
            t += 1
            
        # Z midplane
        z = self.l // 2
        
        # Get electric field
        E_x_list, E_y_list, E_z_list = poisson.get_electric_field()
        
        with open(file_path1, "w") as f1, \
            open(file_path2, "w") as f2, \
            open(file_path3, "w") as f3, \
            open(file_path4, "w") as f4:

        
            # Iterate through the x-y plane with the z midplane
            for x in range(self.l):
                for y in range(self.l):
                    
                    # When not in the middle, due to the monopole
                    if not x == (self.l // 2) and y == (self.l // 2):
                        
                        # Obtain electric fields
                        E_x = E_x_list[x, y, z]
                        E_y = E_y_list[x, y, z]
                        E_z = E_z_list[x, y, z]
                        
                        # Calculate total magnitude
                        E = np.sqrt(E_x**2 + E_y**2 + E_z**2)
                        
                        # Obtain distance
                        r = np.sqrt((x - self.l // 2)**2 + (y - self.l // 2)**2 +\
                                    (z - self.l // 2)**2)
                            
                        # Obtain phi
                        phi = poisson.phi[x, y, z]
                            
                        # Write to files
                        f1.write(f"{phi},{x},{y}\n")
                        f2.write(f"{x},{y},{E_x},{E_y}\n")
                        f3.write(f"{E},{r}\n")
                        f4.write(f"{phi},{r}\n")

    def plot_potential_measurements(self, filename):
        
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
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 
            
        # Create an empty lattice
        phi = np.zeros(shape = (self.l, self.l, self.l))
        
        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Collect the input data
        x = input_data[:, 1].astype(int)
        y = input_data[:, 2].astype(int)
        phi[x, y, self.l // 2] = input_data[:, 0]
        
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the potential
        #plot = plt.imshow(phi[:, :, self.l // 2].T, origin = "lower")
        plot = plt.contour(phi[:, :, self.l // 2].T, levels = 20)
        plt.colorbar(plot, shrink = 0.65)
        ax.set_title(r"Electric Potential of the z-axis midplane", fontsize = 16)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        

    def plot_electric_field_measurements(self, filename):
        
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
                    input_data.append(line.strip("\n").split(","))
                    
        # If text file cannot be found, print error
        except FileNotFoundError:
            print(f"Error: Could not find {filename_path}")
            return 
            
        # Create an empty lattice
        E_x = np.zeros(shape = (self.l, self.l, self.l))
        E_y = np.zeros(shape = (self.l, self.l, self.l))
        
        # Convert input data into a np array
        input_data = np.array(input_data[1:], dtype = float)
        
        # Set z as the midplane
        z = self.l // 2
        
        # Collect the input data
        x = input_data[:, 0].astype(int)
        y = input_data[:, 1].astype(int)
        E_x[x, y, z] = input_data[:, 2]
        E_y[x, y, z] = input_data[:, 3]
        
        # Obtain the midplane
        E_x_midplane = E_x[:, :, z]
        E_y_midplane = E_y[:, :, z]
        
        # Create a meshgrid
        X, Y = np.meshgrid(np.arange(self.l), np.arange(self.l))
        
        # Skip every 3 pixels so the arrows aren't too crowded
        skip = (slice(None, None, 3), slice(None, None, 3))
            
        # Create empty plots
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        
        # Plot the potential
        ax.quiver(X[skip], Y[skip], E_x_midplane.T[skip], E_y_midplane.T[skip])
        ax.set_title(r"Electric Field Vectors $E_x, E_y$", fontsize = 16)
        
        # Save the plots to the plots folder
        save_filename = filename.replace(".txt", "_plot.png")
        save_path = os.path.join(plots_folder, save_filename)
        plt.savefig(save_path, dpi = 300)
        
        # Print message
        print(f"Plots successfully saved to: {save_path}")
        
        # Show final plot
        plt.show()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Poisson")

    # User input parameters
    parser.add_argument("--l", type=int, default=100,
                        help="Lattice size (l x l)")
    # parser.add_argument("--dx", type = float, default = 1, help = "Spatial step")
    # parser.add_argument("--dt", type = float, default = 0.01, help = "Time step")
    parser.add_argument("--mode", type=str, default="ani", choices=["ani", "mea"],
                        help="Animation or measurements")
    parser.add_argument("--steps", type=int, default=100000,
                        help="Number of simulation steps")
    parser.add_argument("--omega", type=float, default=0.1, help="Omega")
    parser.add_argument("--tol", type=float, default=1e-3, help="Tolerance")
    parser.add_argument("--type", type=str, default="e", choices=["e", "m"],
                        help="Electric or magnetic")
    parser.add_argument("--alg", type=str, default="j", choices=["j", "gs", "sor"],
                        help="Algorithms: Jacobi (j), Gauss-Seidel (gs), Over-relaxation (sor)")

    args = parser.parse_args()

    # Pass in parameters to the classes
    sim = Simulation(l=args.l, tolerance=args.tol, omega=args.omega)

    if args.type == "e":

        filename1 = f"p_potential_{args.tol}tol__1.txt"
        filename2 = f"p_electric_{args.tol}tol_1.txt"
        filename3 = f"p_electric_distance_{args.tol}tol_1.txt"
        filename4 = f"p_potential_distance_{args.tol}tol_1.txt"
        sim.measurements(args.alg, filename1, filename2, filename3, filename4)
        sim.plot_potential_measurements(filename1)
        sim.plot_electric_field_measurements(filename2)
        
        
        

    # else:

        # filename = f"ch_free_energy_density_{args.steps}steps_{args.phi}phi_{args.dx}dx_{args.dt}dt_1.txt"
        # sim.measurements(filename, steps = args.steps)
        # sim.plot_measurements(filename)
        
        # Create empty lists to hold temperature, magnetisation and susceptibility    
        x_list = []
        y_list = []
        E_x_list = []
        E_y_list = []
        
# Iterate through input data and append to empty lists
for i in range(0, len(input_data), 4):
    
    # Obtain vlaue from input data
    x = float(input_data[i])
    y = float(input_data[i+1])
    E_x = float(input_data[i+2])
    E_y = float(input_data[i+3])
    
    # Append to empty lists
    x_list.append(x)
    y_list.append(y)
    E_x_list.append(E_x)
    E_y_list.append(E_y)