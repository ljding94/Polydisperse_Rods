from VAE_model import *
import time


def main():
    folder = "../data/data_pack" # for rods data
    ld = 3
    for label in ["20251005_rand_uniform_N20000"]:
        print(f"Processing label: {label}")

        if 0:
            train_and_save_VAE_alone(folder, label, latent_dim=ld, num_epochs=2000)

        if 0:
            train_and_save_generator(folder, label, vae_path=f"{folder}/{label}_vae_state_dict.pt", input_dim=3, latent_dim=ld, num_epochs=300, fine_tune_epochs=300)

        if 0:
            fit_test_data(folder, label, model_path=f"{folder}/{label}_gen_state_dict.pt", latent_dim=ld, input_dim=3, target_loss=1e-4, max_steps=1000, lr=1e-2)

        visualize_LS_fitting_performance(f"{folder}/{label}_test_fits.npz")

        #visualize_param_in_latent_space(f"{folder}/{label}_vae_state_dict.pt", folder, label, latent_dim=ld, save_path=f"{folder}/{label}_latent_distribution.png")

        #plot_loss_curves(folder, label)

        #show_vae_random_reconstructions(folder, label, f"{folder}/{label}_vae_state_dict.pt", latent_dim=ld)

        #show_gen_random_reconstruction(folder, label, f"{folder}/{label}_gen_state_dict.pt", latent_dim=ld)

        #show_infer_random_analysis(folder, label, f"{folder}/{label}_infer_state_dict.pt", latent_dim=ld)

        #show_sample_LS_fitting(folder, label, model_path=f"{folder}/{label}_gen_state_dict.pt", latent_dim=ld, input_dim=3, num_samples=3, num_fits_per_sample=3, target_loss=1e-5, max_steps=3000, lr=1e-2)



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
