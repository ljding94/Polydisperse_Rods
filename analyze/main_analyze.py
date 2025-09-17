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

    if 0:
        plot_PQ_demo()
    if 1:
        # folder = "../data/20250731"
        folder = "../data/20250915"
        run_type = "prec"
        run_num = 0
        pd_type = "uniform"
        N = 10000
        all_system_params = []
        for phi in np.arange(0.03, 0.31, 0.03):
        #for phi in [0.12]:
            #for meanL in np.arange(0.00, 6.10, 1.00):
            for meanL in [4.0]:
                sigmaD = 0.00
                '''
                for sigmaL in np.arange(0.00, 0.201, 0.05):
                    system_params = {
                        "run_type": run_type,
                        "run_num": run_num,
                        "pd_type": pd_type,
                        "N": N,
                        "phi": phi,
                        "meanL": meanL,
                        "sigmaL": sigmaL,
                        "sigmaD": sigmaD,
                    }
                    all_system_params.append(system_params)
                '''
                sigmaL = 0.00
                for sigmaD in np.arange(0.00, 0.201, 0.05):
                    system_params = {
                        "run_type": run_type,
                        "run_num": run_num,
                        "pd_type": pd_type,
                        "N": N,
                        "phi": phi,
                        "meanL": meanL,
                        "sigmaL": sigmaL,
                        "sigmaD": sigmaD,
                    }
                    all_system_params.append(system_params)

        plot_Iq_versus_params(folder, all_system_params)


if __name__ == "__main__":

    main()
