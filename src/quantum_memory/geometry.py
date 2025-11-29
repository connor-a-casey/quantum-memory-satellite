# -*- coding: utf-8 -*-
# Code from arXiv:2402.17752

""" Defines the geometry of the grid where the system of PDE is solved. """

from pde import SphericalSymGrid

ngrid = 1024  # number of grid points
grid = SphericalSymGrid(radius=(0, 1), shape=ngrid)  # generate spatial grid
