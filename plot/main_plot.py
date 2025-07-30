from scattering_plot import *
import numpy as np


def main():
    print("plotting figures for polydisperse hard rods....")

    # 1. Fq interpolation check
    # plot_Fq_interpolate()
    # plot_Iq_xy_fast(L=10.0)  # Example with L/D = 1.0
    plot_Iqxy_single_rod_exact(10, np.pi / 3, np.pi / 3)
    #plot_Iqxy_single_rod_interp(10, np.pi / 3, np.pi / 3)

if __name__ == "__main__":
    main()
