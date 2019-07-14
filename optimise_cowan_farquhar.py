#!/usr/bin/env python

"""
Make a plot of A-lambdaE vs gs for lecture.

Reference:
=========
* Cowan IR, Farquhar GD (1977) Stomatal function in relation to leaf metabolism
  and environment. In: Integration of Activity in the Higher Plant
  (ed. Jennings DH), pp. 471–505. Cambridge University Press, Cambridge.

"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (14.07.2019)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
from scipy.optimize import minimize
from math import fabs
import matplotlib.pyplot as plt

import constants as c
import parameters as p
from farq import FarquharC3

def objective_CF(gsw, *args, **kws):
    """ Find stomatal conductance.

    Parameters
    ----------
    gsw : array
        the gsw value the optimisation routine is varying to maximise An.
    *args : tuple
        series of args required by the photosynthesis model

    """
    # argument unpacking...
    (F, p, par, Cs, Tleaf_K, dleaf, press, lambax) = args

    gsc = gsw / c.GSVGSC
    A = F.photosynthesis(p, Cs=Cs, Tleaf=Tleaf_K, Par=par, vpd=dleaf,
                          gsc=gsc)
    E = c.MOL_2_MMOL * (dleaf * c.KPA_2_PA / press) * gsc * c.GSVGSC
    obj = A - lambdax * E

    # maximising the function
    return obj * -1.

def forward(F, p, par, Cs, Tleaf_K, dleaf, press, lambax, gsw):

    gsc = gsw / c.GSVGSC
    A = F.photosynthesis(p, Cs=Cs, Tleaf=Tleaf_K, Par=par, vpd=dleaf,
                          gsc=gsc)
    E = c.MOL_2_MMOL * (dleaf * c.KPA_2_PA / press) * gsc * c.GSVGSC

    return A - lambdax * E

if __name__ == "__main__":

    par = 1800.0
    Cs = 400.0
    Tleaf = 25.
    Tleaf_K = Tleaf + c.DEG_2_KELVIN
    dleaf = 1.5
    press = 101325.0
    lambdax = 2.0
    F = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=False,
                   gs_model="medlyn")

    N = 50
    gsw_max = np.zeros(3)
    y_max = np.zeros(3)
    y = np.zeros((3,N))
    gsw_vals = np.linspace(0.01, 0.4, N)

    for i, lambdax in enumerate([0.0, 1.0, 3.0]):
        x0 = np.array([0.2])
        bnds = ([(0.001, 0.5)])
        result = minimize(objective_CF, x0, method='SLSQP',
                          args=(F, p, par, Cs, Tleaf_K, dleaf, press, lambdax),
                          bounds=bnds, tol=1e-10)
        gsw_max[i] = result.x[0]


        for j, gsw in enumerate(gsw_vals):
            y[i,j] = forward(F, p, par, Cs, Tleaf_K, dleaf, press, lambdax, gsw)
        y_max[i] = forward(F, p, par, Cs, Tleaf_K, dleaf, press, lambdax,
                           gsw_max[i])

    width = 9
    height = 6
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.02)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['font.size'] = 16
    plt.rcParams['legend.fontsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    ax = fig.add_subplot(111)
    from matplotlib import cm

    import seaborn as sns; sns.set()
    colours = sns.mpl_palette("Set2", 4)

    for i, lambdax in enumerate([0.0, 1.0, 3.0]):

        ax.plot(gsw_vals, y[i,:], label="$\lambda$ = %d" % (lambdax),
                color=colours[i], lw=2)
        if lambdax > 0.0:
            ax.plot(gsw_max[i], y_max[i], marker="o", color=colours[i])

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(r"g$_{\mathrm{s}}$ (mol m$^{-2}$ s$^{-1}$)")
    ax.set_ylabel(r"A - $\lambda$E")
    ax.legend(numpoints=1, ncol=1, frameon=False, loc="best")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ofname = "CF_opt.png"
    fig.savefig(ofname, dpi=300, bbox_inches='tight', pad_inches=0.1)
