""""Hodgkin Huxley model of the cell. 

This code adapted from: 
    https://hodgkin-huxley-tutorial.readthedocs.io/en/latest/_static/Hodgkin%20Huxley.html
"""

import scipy as sp
import pylab as plt
from scipy.integrate import odeint

####################################################################################################
####################################################################################################

class HodgkinHuxley():
    """Full Hodgkin-Huxley Model implemented in Python"""

    C_m  =   1.0 # Membrane capacitance, in uF/cm^2
    
    g_Na = 120.0 # Sodium (Na) maximum conductances, in mS/cm^2
    g_K  =  36.0 # Postassium (K) maximum conductances, in mS/cm^2
    g_L  =   0.3 # Leak maximum conductances, in mS/cm^2

    E_Na =  50.0 # Sodium (Na) Nernst reversal potentials, in mV
    E_K  = -77.0 # Postassium (K) Nernst reversal potentials, in mV

    E_L  = -54.387 # Leak Nernst reversal potentials, in mV

    t = sp.arange(0.0, 450.0, 0.01) # The time to integrate over

    def alpha_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.1*(V+40.0)/(1.0 - sp.exp(-(V+40.0) / 10.0))

    def beta_m(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 4.0*sp.exp(-(V+65.0) / 18.0)

    def alpha_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.07*sp.exp(-(V+65.0) / 20.0)

    def beta_h(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 1.0/(1.0 + sp.exp(-(V+35.0) / 10.0))

    def alpha_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.01*(V+55.0)/(1.0 - sp.exp(-(V+55.0) / 10.0))

    def beta_n(self, V):
        """Channel gating kinetics. Functions of membrane voltage"""
        return 0.125*sp.exp(-(V+65) / 80.0)

    def I_Na(self, V, m, h):
        """Membrane current (in uA/cm^2). Sodium (Na = element name)."""
        return self.g_Na * m**3 * h * (V - self.E_Na)

    def I_K(self, V, n):
        """Membrane current (in uA/cm^2). Potassium (K = element name)."""
        return self.g_K  * n**4 * (V - self.E_K)

    #  Leak
    def I_L(self, V):
        """Membrane current (in uA/cm^2). Leak."""
        return self.g_L * (V - self.E_L)

    def I_inj(self, t):
        """External Current


        |  :return: step up to 10 uA/cm^2 at t>100
        |           step down to 0 uA/cm^2 at t>200
        |           step up to 35 uA/cm^2 at t>300
        |           step down to 0 uA/cm^2 at t>400
        """
        return 10*(t>100) - 10*(t>200) + 35*(t>300) - 35*(t>400)
        
        
def integrate_hh(X, t, HH):
    """Integrate function for the Hodgkin Huxley model."""
    
    V, m, h, n = X

    dVdt = (HH.I_inj(t) - HH.I_Na(V, m, h) - HH.I_K(V, n) - HH.I_L(V)) / HH.C_m

    dmdt = HH.alpha_m(V)*(1.0-m) - HH.beta_m(V)*m
    dhdt = HH.alpha_h(V)*(1.0-h) - HH.beta_h(V)*h
    dndt = HH.alpha_n(V)*(1.0-n) - HH.beta_n(V)*n

    return dVdt, dmdt, dhdt, dndt
    