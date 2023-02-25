# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:21:45 2023

@author: Luis-
"""

from numpy import array
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})


def plot_simple(x, y_h, y_l, gauss_list_ext, xscale, yscale, xlabel, ylabel,
                loc, title, name):
    fig, ax = plt.subplots()
    ax.set_title(title)

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        ax.plot(x, y_h[index],
                label="$\\sigma_{high}$, $l=" + str(l) + ", m=" + str(m) + "$")
        ax.plot(x, y_l[index],
                label="$\\sigma_{low}$,  $l=" + str(l) + ", m=" + str(m) + "$")

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=loc)

    plot_savefig(fig, 'plots/', name, dpi=600)

    return fig, ax


def plot_gauss_solo(x, y, gauss_list_ext, xscale, yscale, xlabel, ylabel,
                    loc, title, name, sharex=False):
    fig, ax = plt.subplots(len(gauss_list_ext), sharex=sharex)
    plt.subplots_adjust(hspace=0)
    ax[0].set_title(title)

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        ax[index].plot(x, y[index],
                       label="$g_{" + str(l) + str(m) + ", \\mathrm{pri}}$")

        if xscale is not None:
            ax[1].set_xscale(xscale)

        ax[index].set_yscale(yscale)
        ax[index].set_ylabel(ylabel)
        ax[index].legend(loc=loc)

    ax[1].set_xlabel(xlabel)

    plot_savefig(fig, 'plots/', name, 600)

    return fig, ax


def plot_gauss_twinx(fig_pri, ax_pri, x, y_h, y_l, gauss_list_ext, ylabel,
                     loc, title, name, axvline=False):
    ax_sec = array([a.twinx() for a in ax_pri.ravel()]).reshape(ax_pri.shape)

    ax_pri[0].set_title(None)
    ax_sec[0].set_title(title)

    for l, m in gauss_list_ext:
        index = gauss_list_ext.index((l, m))

        ax_sec[index].plot(
            x, y_h[index], color='red',
            label="$g_{" + str(l) + str(m) + ", \\mathrm{sec}}, \\sigma=high$")
        ax_sec[index].plot(
            x, y_l[index], color='green',
            label="$g_{" + str(l) + str(m) + ", \\mathrm{sec}}, \\sigma=low$")

        ax_sec[index].set_ylabel(ylabel)
        ax_sec[index].legend(loc=loc)

    if axvline:
        ax_sec[index].axvline(x[0], color='goldenrod', linestyle='dotted')
        ax_sec[1].annotate('$f_1$', (x[0]+5E-7, ax_sec[1].get_ylim()[0]+0.2))

    plot_savefig(fig_pri, 'plots/', name, 600)

    return ax_sec


def plot_savefig(fig, path, name, dpi=600):
    fig.savefig(path + name, dpi=dpi)