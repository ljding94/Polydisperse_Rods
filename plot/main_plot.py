from scattering_plot import *
from data_set_plot import *
from nn_model_plot import *
import numpy as np


def main():
    print("plotting figures for polydisperse hard rods....")

    # 1. Fq interpolation check
    # plot_Fq_interpolate()
    # plot_Iq_xy_fast(L=10.0)  # Example with L/D = 1.0
    #plot_Iqxy_single_rod_exact(10, np.pi / 3, np.pi / 3)
    #plot_Iqxy_single_rod_interp(10, np.pi / 3, np.pi / 3)


    # 1. example Iq and system snapshot (20250929 data)
    plot_example_Iq_and_snapshot()

    # 2. system architecture

    # 3. variation of I(q) versus ,  phi, meanL, sigmaD, sigmaL
    plot_Iq_variation_params()

    # 4. SVD analysis
    plot_SVD_analysis()

    # 5 laten space distribution
    plot_latent_space_distribution()

    # 6. sample generation comparison
    plot_generation_comparison()

    # 7. generation MSE
    plot_generation_MSE()


    # 8 LS fitting performance
    plot_LS_fitting_performance()

    # 9 more distribution (normal and lognormal)
    plot_more_distributions_illustration()

    # 10 normal and lognormal generation MSE

    # 11 normal and lognormal LS fitting performance



if __name__ == "__main__":
    main()
