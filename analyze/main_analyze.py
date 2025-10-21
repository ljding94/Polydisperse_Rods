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



    if 0:
        # folder = "../data/20250731"
        folder = "../data/20250915"
        folder = "../data/20250929"
        folder = "../data/20251001"
        run_type = "prec"
        run_num = 0
        pd_type = "uniform"
        N = 40000
        all_system_params = []
        for phi in np.arange(0.03, 0.31, 0.03):
        #for phi in [0.12]:
            for meanL in np.arange(0.00, 6.10, 1.00):
            #for meanL in [4.0]:
                sigmaD = 0.00
                sigmaL = 0.00
                for sigmaD in np.arange(0.00, 0.251, 0.05):
                #for sigmaD in [0.00]:
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
                sigmaD = 0.00
                for sigmaL in np.arange(0.00, 0.251, 0.05):
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

        label = f"{run_type}_{pd_type}_N{N}"
        #build_Iq_dataset(folder, label, all_system_params)
        dataset_file = f"{folder}/{label}_log10Iq_dataset.npz"
        plot_Iq_versus_params(folder, dataset_file)
        svd_analysis(folder, dataset_file)

    if 1:
        folder = "../data/20251005"
        run_type = "rand"
        run_num = 0
        pd_type = "uniform"
        N = 20000
        all_system_params = []
        for run_num in range(5120):
            system_params = {
                "run_type": run_type,
                "run_num": run_num,
                "pd_type": pd_type,
                "N": N,
            }
            all_system_params.append(system_params)
        label = f"{run_type}_{pd_type}_N{N:.0f}"
        build_Iq_dataset(folder, label, all_system_params)
        dataset_file = f"{folder}/{label}_log10Iq_dataset.npz"
        plot_Iq_versus_params(folder, dataset_file)
        svd_analysis(folder, dataset_file)

if __name__ == "__main__":

    main()
