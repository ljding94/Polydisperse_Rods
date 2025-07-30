from Iq_analyze import *

# Add parent directory to path so we can import from there
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "simulation"))
#from IQ import build_FQ_interpolator


def main():

    print("Analyzing scattering data...")

    if 0:
        # examine the FQ calculation
        plot_Iqxy_single_rod_exact(6, np.pi / 3, np.pi / 3)
        #plot_Iqxy_single_rod_interp(6, np.pi / 3, np.pi / 3)

    if 1:
        filename = "../data_local/data_pool/stats_sample_Iq_prec_run0_uniform_N20_phi0.10_mean_ld4.00_sigma0.10.csv"
        plot_sample_Iq(filename)


if __name__ == "__main__":

    main()





