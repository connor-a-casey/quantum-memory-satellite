# -*- coding: utf-8 -*-
# Code from arXiv:2402.17752

"""
Defines a class of PDE that model the dynamics of the quantum memory protocol.

In this model, the expectation values of the collective spin operators for the
alkali and noble-gas spins are calculated taking the effects of diffusion into
account, using the method of lines for spatial discretization. In this way, the
efficiency of the spin-exchange mechanism between the spins can be estimated.

The expectation value of each spin operator is represented by a ScalarField
object, and the relevant physical parameters are parameters of this new class.
The rate of evolution is calculated by a class method, and compiled with numba.

"""

import numpy as np
import numba as nb
from pde import FieldCollection, PDEBase, ScalarField
from .geometry import grid


class CollectiveSpinsPDE(PDEBase):
    """
    A system of PDE describing the dynamics of the quantum memory protocol.

    :get_initial_state: initializes scalar fields representing spin operators
    :magnetic_detuning: computes the externally controllable detuning at time t
    :evolution_rate: computes the rhs of the PDE system at time t

    physical parameters are provided as class variables (described in __init__)
    in nondimensional units where both the alkali optical coherence decay rate
    (5.96 x 2π MHz) and the vapor cell radius (1 cm) correspond to unity.
    """

    def __init__(self, bca="dirichlet", bcb="neumann", J=1.85e-5, gamma_k=0,
                 gamma_s=3.11e-7, na=5.2e14, nb=5.4e19, delta_s=0,
                 delta_k=1.11e-3, Da=9.35e-9, Db=1.87e-8, t0=1e4, tdark=7.5e4):
        super().__init__()
        self.complex_valued = True
        self.bca = bca  # alkali spins boundary condition
        self.bcb = bcb  # noble-gas spins boundary condition
        self.gamma_k = gamma_k  # decay rate for noble-gas spin coherence
        self.gamma_s = gamma_s  # decay rate for alkali spin coherence
        self.na = na  # alkali-metal gas density
        self.nb = nb  # noble-gas density
        self.delta_s = delta_s   # two-photon detuning (alkali atoms)
        self.delta_k = delta_k   # two-photon detuning (noble-gas spins)
        self.Da = Da  # diffusion coefficient for alkali-metal spins
        self.Db = Db  # diffusion coefficient for noble-gas spins
        self.J = J  # alkali-noble gas spin-exchange coupling rate
        self.t0 = t0  # starting time of the memory protocol
        self.tdark = tdark  # memory time in the dark
        # compute the transfer pulse time tpulse from provided parameters
        diff = np.abs((1j*self.gamma_s)*(1j*self.gamma_s)/4)
        Jtilde = np.sqrt(J*J+diff)
        self.tpulse = (np.pi*Jtilde-self.gamma_s)/(2*Jtilde*Jtilde)
        self.tr = 2*self.tpulse+self.tdark+self.t0  # total memory time

    def get_initial_state(self, grid):
        """
        Generate the initial state of the system.

        :grid: spatial grid, defined in geometry.py
        :return: scalar fields with values of the collective spin operators
        """
        S = ScalarField(grid, 1, label=r"$\langle \hat{\mathcal{S}}^{\dag}  \
                        \hat{\mathcal{S}}  \rangle$", dtype=complex)
        K = ScalarField(grid, 0, label=r"$\langle \hat{\mathcal{K}}^{\dag} \
                        \hat{\mathcal{K}}  \rangle$", dtype=complex)

        return FieldCollection([S, K])

    def magnetic_detuning(self, t):
        """
        Compute magnetic detuning throughout the protocol for square control.

        :t: simulation time
        :return: delta_k at time t
        """
        tr = 2*self.tpulse+self.tdark+self.t0
        t1 = self.tpulse + self.t0
        t2 = self.tdark + self.t0 + self.tpulse

        return self.delta_k if (t < self.t0) or (t > t1 and t < t2) or \
            (t > tr) else 0

    def evolution_rate(self, state, t=0):
        """
        Calculate the rhs of the pde at a given time.

        :state: current values of the collective spin operators
        :t: simulation time
        :return: scalar fields with evolution rate of the spin operators
        """
        S, K = state
        delta_k = self.magnetic_detuning(t)

        dS_dt = -(self.gamma_s + 1j*self.delta_s)*S + \
            self.Da*S.laplace(bc=self.bca) - 1j*self.J*K
        dK_dt = -(self.gamma_k + 1j*delta_k)*K + \
            self.Db*K.laplace(bc=self.bcb) - 1j*self.J*S

        return FieldCollection([dS_dt, dK_dt])

    def _make_pde_rhs_numba(self, state):
        """
        Create a numba-compiled implementation of the pde rhs for speed.

        :state: current values of the collective spin operators
        :return: numba-compiled implementation of the evolution rate
        """
        laplace_a = state.grid.make_operator("laplace", bc=self.bca)
        laplace_b = state.grid.make_operator("laplace", bc=self.bcb)
        t0 = self.t0
        J = self.J
        tpulse = self.tpulse
        tdark = self.tdark
        tr = 2*self.tpulse+self.tdark+self.t0
        gamma_s = self.gamma_s
        delta_s = self.delta_s
        magnetic_detuning = self.delta_k
        Da = self.Da
        Db = self.Db
        gamma_k = self.gamma_k

        @nb.njit
        def pde_rhs(data, t):
            S = data[0]
            K = data[1]
            rhs = np.empty_like(data)

            # Magnetic Detuning
            t1 = tpulse + t0
            t2 = tdark + t0 + tpulse
            if (t < t0) or (t > t1 and t < t2) or (t > tr):
                delta_k = magnetic_detuning
            else:
                delta_k = 0

            # PDE RHS
            rhs[0] = -(gamma_s + 1j*delta_s)*S + Da*laplace_a(S) - 1j*J*K
            rhs[1] = -(gamma_k + 1j*delta_k)*K + Db*laplace_b(K) - 1j*J*S

            return rhs

        return pde_rhs
