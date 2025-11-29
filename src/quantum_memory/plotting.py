# -*- coding: utf-8 -*-
# Code from arXiv:2402.17752

""" Defines functions for plotting the evolution of the spin operators. """

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import csv
from matplotlib import rc, rcParams
from .dynamics import CollectiveSpinsPDE
from pde import FileStorage, ScalarField
from .geometry import ngrid

# figure style, using the scienceplots package
plt.rcParams.update({'figure.dpi': '600'})
plt.style.use(['science', 'notebook', 'grid'])
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


def plot_kymograph(read_filename, sim, operator='S', save_filename='kymo.pdf'):
    """
    Plot a kymograph showing the evolution of the collective spin operators.

    :read_filename: file containing operator values in each point and time
    :sim: instance of the CollectiveSpinsPDE() class
    :operator: alkali ('S') or noble-gas ('K') collective spin operator
    :save_filename: kymograph is saved with this parameter as a filename
    :return: none
    """
    reader = FileStorage(read_filename, write_mode="read_only")

    field = []  # operator expectation values through time and space
    t = []  # array of simulation times

    for time, collection in reader.items():
        S, K = collection.fields

        if operator == 'S':
            field.append(np.abs(S.data)*np.abs(S.data))

        elif operator == 'K':
            field.append(np.abs(K.data)*np.abs(K.data))

        t.append(time)

    """ Spherical Grid """
    npoints = ngrid

    r = []
    for i in range(npoints):
        r.append((i+0.5)/npoints)

    X, Y = np.meshgrid(t, r)

    if operator == 'S':
        cs = plt.contourf(X, Y, np.transpose(field), levels=100, cmap='OrRd', vmin=0, vmax=1.0)
        plt.title(r'$\langle \hat{\mathcal{S}}^{\dag} \hat{\mathcal{S}}' +
                  r'\rangle $', fontsize=22, pad=20)

    elif operator == 'K':
        cs = plt.contourf(X, Y, np.transpose(field), levels=80, cmap='PuBu', vmin=0, vmax=1.0)
        plt.title(r'$\langle \hat{\mathcal{K}}^{\dag} \hat{\mathcal{K}}' +
                  r'\rangle $', fontsize=22, pad=20)

    plt.ylabel(r'$r/R$', fontsize=20)

    t_pi = sim.tpulse + sim.t0
    t_d = sim.tpulse + sim.t0 + sim.tdark
    plt.xticks([sim.t0, t_pi, t_d, sim.tr],
               [r'$0$', r'$T_\pi$', r'$T_D + T_\pi$', r'$T_R$'], fontsize=18)

    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.minorticks_off()
    plt.colorbar(cs, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig(save_filename)
    plt.show()


def plot_magnitude(read_filename, sim, operator, save_filename='mag.pdf'):
    """
    Plot the evolution of the collective spin operator magnitudes.

    :read_filename: file containing operator magnitudes across time
    :sim: instance of the CollectiveSpinsPDE class
    :operator: alkali ('S') or noble-gas ('K') collective spin operator
    :save_filename: plot is saved with this parameter as a filename
    :return: none
    """
    t = []
    S = []
    K = []

    with open(read_filename, 'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter=',')
        headers = next(magnitudes)  # ignore file header
        for row in magnitudes:
            try:
                t.append(float(row[0]))
                S.append(float(row[1]))
                K.append(float(row[2]))
            except ValueError:
                pass

    S = np.array(S)
    t = np.array(t)

    if operator == 'S':
        plt.plot(t, S, color='crimson')
        plt.ylabel(r'$\langle \hat{\mathcal{S}}^{\dag}' +
                   r'\hat{\mathcal{S}} \rangle $', fontsize=22)
        plt.fill_between(t, S, alpha=0.25, interpolate=True, color='orangered')
        plt.ylim(0, 1)
        plt.grid(visible=False)

    elif operator == 'K':
        plt.plot(t, K, color='blue')
        plt.ylabel(r'$\langle \hat{\mathcal{K}}^{\dag}' +
                   r'\hat{\mathcal{K}} \rangle $', fontsize=22)
        plt.fill_between(t, K, alpha=0.25, interpolate=True, color='blue')
        plt.grid(visible=False)

    t_min = -sim.tpulse/3  # cosmetic
    t_prime = sim.tpulse + sim.t0
    t_d = sim.tpulse + sim.t0 + sim.tdark

    plt.xticks([t_min, sim.t0, t_prime, t_d, sim.tr],
               [r'$-\infty$', '$0$', r"$T'$", r"$T_D + T'$", r'$T_R$'],
               fontsize=18)

    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=18)
    plt.minorticks_off()
    plt.savefig(save_filename)
    plt.show()
