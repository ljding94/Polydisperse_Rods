from Iq_analyze import *

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
# from IQ import build_FQ_interpolator


def main():

    print("Analyzing scattering data...")

    if 0:
        # examine the FQ calculation
        plot_Iqxy_single_rod_exact(6, np.pi / 3, np.pi / 3)
        # plot_Iqxy_single_rod_interp(6, np.pi / 3, np.pi / 3)

    if 0:
        # filename = "../data_local/data_pool/stats_sample_Iq_prec_run0_uniform_N20_phi0.10_mean_ld4.00_sigma0.10.csv"
        filename = "../data_local/data_pool/stats_sample_Iq_prec_run0_uniform_N100_phi0.10_mean_ld4.00_sigma0.10.csv"
        plot_sample_Iq(filename)

    if 1:
        plot_PQ_demo()
    if 0:
        folder = "../data/20250731"
        phis = np.arange(0.05, 0.31, 0.05)
        mean_lds = np.arange(1.00, 4.10, 0.40)
        sigmas = np.arange(0.00, 0.21, 0.02)
        all_system_params = []
        for phi in phis:
            for mean_ld in mean_lds:
                for sigma in sigmas:
                    system_params = {
                        "run_type": "prec",
                        "run_num": 0,
                        "pd_type": "uniform",
                        "N": 10000,
                        "phi": phi,
                        "mean_ld": mean_ld,
                        "sigma": sigma,
                    }
                    all_system_params.append(system_params)

        plot_Iq_versus_params(folder, all_system_params)



if __name__ == "__main__":

    main()
