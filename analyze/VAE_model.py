# all NN function for VAE
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


def denormalize_generated_Iq(folder, label, log10Iq_norm, params_norm):
    data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    log10Iq_mean = data_stats["mean_log10Iq"]
    log10Iq_std = data_stats["std_log10Iq"]
    params_name = data_stats["params_name"]
    params_mean = data_stats["mean_params"]
    params_std = data_stats["std_params"]
    log10Iq = log10Iq_norm * log10Iq_std + log10Iq_mean  # denormalize)

    if "sigmaL" in params_name:
        sigmaL_index = list(params_name).index("sigmaL")
        params_mean = np.delete(params_mean, sigmaL_index, axis=0)
        params_std = np.delete(params_std, sigmaL_index, axis=0)
        params_name = np.delete(params_name, sigmaL_index)
    params = params_norm * params_std + params_mean  # denormalize parameters
    return log10Iq, params


def normalize_Iq(folder, label, log10Iq, params):
    data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    log10Iq_mean = data_stats["mean_log10Iq"]
    log10Iq_std = data_stats["std_log10Iq"]
    params_name = data_stats["params_name"]
    params_mean = data_stats["mean_params"]
    params_std = data_stats["std_params"]
    log10Iq_norm = (log10Iq - log10Iq_mean) / log10Iq_std  # normalize to mean=0, std=1 for each q point
    # we need to remove sigmaL since it's always 0
    if "sigmaL" in params_name:
        sigmaL_index = list(params_name).index("sigmaL")
        params_mean = np.delete(params_mean, sigmaL_index, axis=0)
        params_std = np.delete(params_std, sigmaL_index, axis=0)
        params_name = np.delete(params_name, sigmaL_index)
    params_norm = (params - params_mean) / params_std  # normalize parameters
    return log10Iq_norm, params_norm

# --------------------------
# Data Loading Functions
# --------------------------


class logIqDataset(Dataset):
    def __init__(self, folder, label, data_type):
        # start with simple parsing
        # get raw vol data
        Iq_data = np.load(f"{folder}/{label}_log10Iq_dataset_{data_type}.npz")
        print(Iq_data.keys())
        print("Iq_data[\"params_name\"]", Iq_data["params_name"])
        # get data stats
        data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
        print(data_stats.keys())
        self.log10Iq_mean = data_stats["mean_log10Iq"]
        self.log10Iq_std = data_stats["std_log10Iq"]
        params_name = data_stats["params_name"]
        self.params_mean = data_stats["mean_params"]
        self.params_std = data_stats["std_params"]

        self.q = Iq_data["q"]
        self.log10Iq = (Iq_data["all_log10Iq"] - self.log10Iq_mean) / self.log10Iq_std  # normalize to mean=0, std=1 for each q point
        Iq_params = (Iq_data["all_params"] - self.params_mean) / self.params_std  # normalize parameters

        self.params = Iq_params
        self.params_name = Iq_data["params_name"]

        # we need to remove sigmaL since it's always 0
        if "sigmaL" in self.params_name:
            sigmaL_index = list(self.params_name).index("sigmaL")
            self.params = np.delete(self.params, sigmaL_index, axis=1)
            self.params_name = np.delete(self.params_name, sigmaL_index)
            print("Removed sigmaL from params")

        print("Dataset initialized:")
        print("self.params_name", self.params_name)
        print("self.log10Iq.shape, self.q.shape, self.params.shape")
        print(self.log10Iq.shape, self.q.shape, self.params.shape)

    def __len__(self):
        return len(self.log10Iq)

    def __getitem__(self, idx):
        log10Iq = self.log10Iq[idx]
        q = self.q
        params = self.params[idx]
        log10Iq = torch.from_numpy(log10Iq).float().unsqueeze(0)  # add channel for conv layer
        q = torch.tensor(q, dtype=torch.float32)
        params = torch.tensor(params, dtype=torch.float32)

        return log10Iq, q, params


def create_dataloader(folder, label, data_type, batch_size=32, shuffle=True, transform=None):
    dataset = logIqDataset(folder, label, data_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset


# --------------------------
# Model Definitions
# --------------------------
"""
Input (100,) ──► Encoder ──► z  ──► Decoder ──► generator f(y) ──►  x̂
                          │ │
                    ŷ ────  └──► inferrer g(z) ──► ŷ
Loss =  MSE(x̂,x)  +  MSE(ŷ,y)
"""


# ---------- Encoder ----------
class Encoder(nn.Module):
    def __init__(self, input_dim=100, latent_dim=3):
        super().__init__()
        # 1D convolution layers
        self.conv = nn.Sequential(
            nn.Conv1d(1, 30, kernel_size=9, stride=2, padding=4),  # (100,) -> (50,)
            #nn.BatchNorm1d(30), # for our small dataset, batchnorm not helping
            nn.ReLU(),
            nn.Conv1d(30, 60, kernel_size=9, stride=2, padding=4),  # (50,) -> (25,)
            #nn.BatchNorm1d(60),
            nn.ReLU(),
        )
        # Calculate flattened size after conv layers
        self.flatten_dim = 60 * 25  # 50 channels, each of size 25 after conv

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)  # Apply 1D convolution layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, flatten_dim)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


# ---------- Decoder ----------
class Decoder(nn.Module):
    def __init__(self, latent_dim=3, output_dim=100):
        super().__init__()
        # Calculate the size needed to match encoder's flatten_dim
        self.flatten_dim = 60 * 25  # Should match encoder's flatten_dim
        self.fc = nn.Linear(latent_dim, self.flatten_dim)
        # Transpose convolution layers (reverse of encoder)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(60, 30, kernel_size=9, stride=2, padding=4, output_padding=1),  # (25,) -> (50,)
            #nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.ConvTranspose1d(30, 1, kernel_size=9, stride=2, padding=4, output_padding=1),  # (50,) -> (100,)
        )

    def forward(self, z):  # z: (..., latent_dim)
        orig_shape = z.shape[:-1]  # (100, B)
        x = z.reshape(-1, z.size(-1))  # (100*B, latent_dim)
        x = self.fc(x)  # (100*B, 1250)
        x = x.view(-1, 60, 25)  # (100*B, 50, 25)
        x = self.deconv(x)  # (100*B, 1, 100)
        x = x.squeeze(1)  # (100*B, 100)
        return x.view(*orig_shape, -1)  # (100, B, 100)


# ---------- VAE ----------
class VAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=6):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    @staticmethod
    def reparameterise(mu, logvar):
        std, eps = (0.5 * logvar).exp(), torch.randn_like(logvar)
        return mu + eps * std

    def forward(self, x, u=None, *, deterministic=False):
        mu, logvar = self.encoder(x)
        epsilons = torch.randn(100, *mu.shape, device=mu.device) # hardcode 100 samples
        z_samples = mu.unsqueeze(0) + epsilons * (0.5 * logvar).exp().unsqueeze(0)
        recons = self.decoder(z_samples)
        recon_avg = recons.mean(dim=0)
        return recon_avg, mu, logvar


class ConverterP2L(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 9),
            #nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Linear(9, 9),
            #nn.BatchNorm1d(9),
            nn.ReLU(),
        )
        # Branch for mu and logcar
        self.fc_mu = nn.Linear(9, latent_dim)
        self.fc_logvar = nn.Linear(9, latent_dim)

    def forward(self, x):
        h = self.shared(x)  # (B, input_dim) → (B, 9)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConverterL2P(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 9),
            #nn.BatchNorm1d(9),
            nn.ReLU(),
            nn.Linear(9, 9),
            nn.ReLU(),
            nn.Linear(9, output_dim),
        )

    def forward(self, z):
        return self.fc(z)  # (B, latent_dim) → (B, output_dim)


class Generator(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, output_dim=100):
        super().__init__()
        self.cvtp2l = ConverterP2L(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x):
        mu, logvar = self.cvtp2l(x)
        epsilons = torch.randn(100, *mu.shape, device=mu.device)  # (100, B, latent_dim)
        z_samples = mu.unsqueeze(0) + epsilons * (0.5 * logvar).exp().unsqueeze(0)  # (100, B, latent_dim)
        recons = self.decoder(z_samples)  # (100, B, input_dim)
        recon_avg = recons.mean(dim=0)  # Average over samples: (B, input_dim)
        return recon_avg, mu, logvar


class Inferrer(nn.Module):
    def __init__(self, input_dim=100, latent_dim=3, output_dim=2):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)  # Use same encoder as VAE
        self.cvtl2p = ConverterL2P(latent_dim, output_dim)  # Convert latent to parameters

    def forward(self, z):
        mu, logvar = self.encoder(z)
        epsilons = torch.randn(100, *mu.shape, device=mu.device) # (100, B, latent_dim)
        z_samples = mu.unsqueeze(0) + epsilons * (0.5 * logvar).exp().unsqueeze(0)
        pred = self.cvtl2p(z_samples)  # (100, B, output_dim)
        pred_avg = pred.mean(dim=0)
        return pred_avg, mu, logvar


def vae_ensemble_loss(x, vae_model):
    # 1. get latent code from encoder
    # mu, logvar = vae_model.encoder(x)

    # 2 sample 100 epsilons for calculate the ensemble average of reconstruction
    # epsilons = torch.randn(100, *mu.shape, device=mu.device)  # (100, B, latent_dim)
    # z_samples = mu.unsqueeze(0) + epsilons * (0.5 * logvar).exp().unsqueeze(0)  # (100, B, latent_dim)
    # reconstructions = vae_model.decoder(z_samples)  # (100, B, input_dim)
    recon_avg, _, _ = vae_model(x)
    # 3. calculate the reconstruction loss
    recon_loss = F.mse_loss(recon_avg.unsqueeze(1), x, reduction="mean")
    return recon_loss


# x are scattering Iq, y are the parameters
def generator_loss(p, Iq, gen_model):
    # 1 get laten code from ConverterP2L
    recon_avg, mu, logvar = gen_model(p)
    recon_loss = F.mse_loss(recon_avg.unsqueeze(1), Iq, reduction="mean")
    # Add regularization loss for smoothness
    # recon_flat = recon_avg.view(-1, recon_avg.size(-1))  # (B, 100)
    # diff = recon_flat[:, 1:] - recon_flat[:, :-1]  # First-order differences
    # smooth_loss = torch.mean(diff**2)  # L2 penalty on differences
    # recon_loss = recon_loss + 0.01 * smooth_loss  # Add with weight factor
    return recon_loss


def inferrer_loss(p, Iq, inf_model):
    pred_avg, _, _ = inf_model(Iq)
    pred_loss = F.mse_loss(pred_avg, p, reduction="mean")
    return pred_loss


def train_and_save_VAE_alone(folder: str, label: str, latent_dim: int = 3, batch_size: int = 32, num_epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure save directory exists
    os.makedirs(folder, exist_ok=True)

    # Data loaders
    train_loader, _ = create_dataloader(folder, label, "train", batch_size=batch_size, shuffle=True)
    test_loader, _ = create_dataloader(folder, label, "test", batch_size=batch_size, shuffle=False)
    print(f"Training VAE on {len(train_loader.dataset)} samples, validating on {len(test_loader.dataset)} samples.")

    # Model, optimizer, scheduler
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler will anneal LR from `lr` → `lr*lr_min_mult` over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)  # full period is the training run  # final learning rate
    train_losses, test_losses = [], []
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        # log10Iq, q, params  in tran_loader
        for x, _, _ in train_loader:
            x = x.to(device)
            recon_loss = vae_ensemble_loss(x, model)
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            bs = x.size(0)
            total_train_loss += recon_loss.item() * bs
        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # Evaluate on test set
        model.eval()
        total_test_loss = 0.0
        for x_test, _, _ in test_loader:
            x_test = x_test.to(device)
            recon_loss_test = vae_ensemble_loss(x_test, model)
            bs = x_test.size(0)
            total_test_loss += recon_loss_test.item() * bs

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.8f} " f"| test_loss={avg_test_loss:.8f} ")

    # Save model state dict and loss histories
    state_path = os.path.join(folder, f"{label}_vae_state_dict.pt")
    torch.save(model.state_dict(), state_path)
    np.savez(os.path.join(folder, f"{label}_vae_losses.npz"), train_losses=np.array(train_losses), test_losses=np.array(test_losses))

    print(f"Saved model state to {state_path}")
    print(f"Saved train/test losses to {folder}")

    return model, train_losses, test_losses


def train_and_save_generator(
    folder: str,
    label: str,
    vae_path: str,
    input_dim: int = 3,
    latent_dim: int = 3,
    batch_size: int = 32,
    num_epochs: int = 100,
    fine_tune_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_loader, _ = create_dataloader(folder, label, "train", batch_size=batch_size, shuffle=True)
    test_loader, _ = create_dataloader(folder, label, "test", batch_size=batch_size, shuffle=False)
    print(f"Training generator on {len(train_loader.dataset)} samples, validating on {len(test_loader.dataset)} samples.")

    # read vae and model
    vae_model = VAE(latent_dim=latent_dim)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.to(device)

    # initialize cvtp2l model
    # initialize generator model
    gen_model = Generator(input_dim=input_dim, latent_dim=latent_dim).to(device)
    gen_model.decoder = vae_model.decoder  # share the decoder
    # freeze decoder parameters
    for param in gen_model.decoder.parameters():
        param.requires_grad = False

    # Ensure save directory exists
    os.makedirs(folder, exist_ok=True)

    # train cvtp2l
    optimizer = optim.Adam(gen_model.cvtp2l.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)  # full period is the training run  # final learning rate
    train_losses, test_losses = [], []
    # Training loop
    for epoch in range(num_epochs):
        gen_model.cvtp2l.train()
        total_train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]
        # log10Iq, q, params  in tran_loader
        for Iq, _, p in train_loader:
            Iq, p = Iq.to(device), p.to(device)
            gen_loss = generator_loss(p, Iq, gen_model)
            optimizer.zero_grad()
            gen_loss.backward()
            optimizer.step()

            bs = Iq.size(0)
            total_train_loss += gen_loss.item() * bs

        scheduler.step()  # ← step the LR scheduler once per epoch
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        # Evaluate on test set
        gen_model.eval()
        total_test_loss = 0.0
        for Iq_test, _, p_test in test_loader:
            Iq_test, p_test = Iq_test.to(device), p_test.to(device)
            gen_loss_test = generator_loss(p_test, Iq_test, gen_model)
            bs = Iq_test.size(0)
            total_test_loss += gen_loss_test.item() * bs
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.8f} " f"| test_loss={avg_test_loss:.8f} ")

    # fine tuning by unfreezing decoder
    for param in gen_model.decoder.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(list(gen_model.parameters()), lr=lr * 0.1, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.001)  # full period is the training run  # final learning rate

    fine_tune_train_losses, fine_tune_test_losses = [], []
    for epoch in range(fine_tune_epochs):
        gen_model.train()
        total_train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        # log10Iq, q, params  in tran_loader
        for Iq, _, p in train_loader:
            Iq, p = Iq.to(device), p.to(device)
            gen_loss = generator_loss(p, Iq, gen_model)
            optimizer.zero_grad()
            gen_loss.backward()
            optimizer.step()

            bs = Iq.size(0)
            total_train_loss += gen_loss.item() * bs
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        fine_tune_train_losses.append(avg_train_loss)

        # Evaluate on test set
        gen_model.eval()
        total_test_loss = 0.0
        for Iq_test, _, p_test in test_loader:
            Iq_test, p_test = Iq_test.to(device), p_test.to(device)
            gen_loss_test = generator_loss(p_test, Iq_test, gen_model)
            bs = Iq_test.size(0)
            total_test_loss += gen_loss_test.item() * bs
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        fine_tune_test_losses.append(avg_test_loss)
        print(f"[Fine-tune] Epoch {epoch+1}/{fine_tune_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.8f} " f"| test_loss={avg_test_loss:.8f} ")

    # save train loss, test loss, and fine tune loss to npz
    state_path = os.path.join(folder, f"{label}_gen_state_dict.pt")
    torch.save(gen_model.state_dict(), state_path)

    np.savez(
        os.path.join(folder, f"{label}_gen_losses.npz"),
        train_losses=np.array(train_losses),
        test_losses=np.array(test_losses),
        fine_tune_train_losses=np.array(fine_tune_train_losses),
        fine_tune_test_losses=np.array(fine_tune_test_losses),
    )

def train_and_save_inferrer(
    folder: str,
    label: str,
    vae_path: str,
    input_dim: int = 100,
    latent_dim: int = 3,
    output_dim: int = 2,
    batch_size: int = 32,
    num_epochs: int = 100,
    fine_tune_epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_loader, _ = create_dataloader(folder, label, "train", batch_size=batch_size, shuffle=True)
    test_loader, _ = create_dataloader(folder, label, "test", batch_size=batch_size, shuffle=False)
    print(f"Training inferrer on {len(train_loader.dataset)} samples, validating on {len(test_loader.dataset)} samples.")

    # read vae and model
    vae_model = VAE(latent_dim=latent_dim)
    vae_model.load_state_dict(torch.load(vae_path, map_location=device))
    vae_model.to(device)

    # initialize inferrer model
    inf_model = Inferrer(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim).to(device)
    inf_model.encoder = vae_model.encoder  # share the encoder
    # freeze encoder parameters
    for param in inf_model.encoder.parameters():
        param.requires_grad = False

    # Ensure save directory exists
    os.makedirs(folder, exist_ok=True)

    # train cvtl2p
    optimizer = optim.Adam(inf_model.cvtl2p.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)  # full period is the training run  # final learning rate
    train_losses, test_losses = [], []
    # Training loop
    for epoch in range(num_epochs):
        inf_model.cvtl2p.train()
        total_train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]
        # log10Iq, q, params  in tran_loader
        for Iq, _, p in train_loader:
            Iq, p = Iq.to(device), p.to(device)
            inf_loss = inferrer_loss(p, Iq, inf_model)
            optimizer.zero_grad()
            inf_loss.backward()
            optimizer.step()

            bs = Iq.size(0)
            total_train_loss += inf_loss.item() * bs

        scheduler.step()  # ← step the LR scheduler once per epoch
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        # Evaluate on test set
        inf_model.eval()
        total_test_loss = 0.0
        for Iq_test, _, p_test in test_loader:
            Iq_test, p_test = Iq_test.to(device), p_test.to(device)
            inf_loss_test = inferrer_loss(p_test, Iq_test, inf_model)
            bs = Iq_test.size(0)
            total_test_loss += inf_loss_test.item() * bs
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch+1}/{num_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.8f} " f"| test_loss={avg_test_loss:.8f} ")


    # fine tuning by unfreezing decoder
    for param in inf_model.encoder.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(list(inf_model.parameters()), lr=lr * 0.1, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.001)  # full period is the training run  # final learning rate
    fine_tune_train_losses, fine_tune_test_losses = [], []
    for epoch in range(fine_tune_epochs):
        inf_model.train()
        total_train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]

        # log10Iq, q, params  in tran_loader
        for Iq, _, p in train_loader:
            Iq, p = Iq.to(device), p.to(device)
            inf_loss = inferrer_loss(p, Iq, inf_model)
            optimizer.zero_grad()
            inf_loss.backward()
            optimizer.step()

            bs = Iq.size(0)
            total_train_loss += inf_loss.item() * bs
        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        fine_tune_train_losses.append(avg_train_loss)

        # Evaluate on test set
        inf_model.eval()
        total_test_loss = 0.0
        for Iq_test, _, p_test in test_loader:
            Iq_test, p_test = Iq_test.to(device), p_test.to(device)
            inf_loss_test = inferrer_loss(p_test, Iq_test, inf_model)
            bs = Iq_test.size(0)
            total_test_loss += inf_loss_test.item() * bs
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        fine_tune_test_losses.append(avg_test_loss)
        print(f"[Fine-tune] Epoch {epoch+1}/{fine_tune_epochs} " f"| LR={current_lr:.2e} " f"| train_loss={avg_train_loss:.8f} " f"| test_loss={avg_test_loss:.8f} ")
    # save train loss, test loss, and fine tune loss to npz
    state_path = os.path.join(folder, f"{label}_infer_state_dict.pt")
    torch.save(inf_model.state_dict(), state_path)
    np.savez(
        os.path.join(folder, f"{label}_infer_losses.npz"),
        train_losses=np.array(train_losses),
        test_losses=np.array(test_losses),
        fine_tune_train_losses=np.array(fine_tune_train_losses),
        fine_tune_test_losses=np.array(fine_tune_test_losses),
    )



def plot_loss_curves(folder: str, label: str, show: bool = True):
    """
    -----------------------------------------------------------
    Plot train / test loss versus *epoch* for VAE, Generator, and Inferrer models.
    Parameters
    ----------
    folder       : directory that contains the saved .npz files
    label        : label for the dataset
    show         : whether to call plt.show()  (set False in scripts)
    -----------------------------------------------------------
    """
    vae_losses_path = os.path.join(folder, f"{label}_vae_losses.npz")
    gen_losses_path = os.path.join(folder, f"{label}_gen_losses.npz")
    inf_losses_path = os.path.join(folder, f"{label}_inf_losses.npz")

    # Load VAE losses
    if not os.path.exists(vae_losses_path):
        print(f"Could not find VAE losses file: {vae_losses_path}. Using empty arrays.")
        vae_train_losses = np.array([])
        vae_test_losses = np.array([])
    else:
        vae_losses_data = np.load(vae_losses_path)
        vae_train_losses = vae_losses_data["train_losses"]
        vae_test_losses = vae_losses_data["test_losses"]
    vae_epochs = np.arange(1, len(vae_train_losses) + 1) if len(vae_train_losses) > 0 else np.array([])

    # Load Generator losses
    if not os.path.exists(gen_losses_path):
        print(f"Could not find Generator losses file: {gen_losses_path}. Using empty arrays.")
        gen_train_losses = np.array([])
        gen_test_losses = np.array([])
        gen_fine_tune_train_losses = np.array([])
        gen_fine_tune_test_losses = np.array([])
    else:
        gen_losses_data = np.load(gen_losses_path)
        gen_train_losses = gen_losses_data["train_losses"]
        gen_test_losses = gen_losses_data["test_losses"]
        gen_fine_tune_train_losses = gen_losses_data["fine_tune_train_losses"]
        gen_fine_tune_test_losses = gen_losses_data["fine_tune_test_losses"]
    gen_epochs = np.arange(1, len(gen_train_losses) + 1) if len(gen_train_losses) > 0 else np.array([])
    gen_fine_tune_epochs = np.arange(len(gen_train_losses) + 1, len(gen_train_losses) + len(gen_fine_tune_train_losses) + 1) if len(gen_fine_tune_train_losses) > 0 else np.array([])

    # Load Inferrer losses
    if not os.path.exists(inf_losses_path):
        print(f"Could not find Inferrer losses file: {inf_losses_path}. Using empty arrays.")
        inf_train_losses = np.array([])
        inf_test_losses = np.array([])
        inf_fine_tune_train_losses = np.array([])
        inf_fine_tune_test_losses = np.array([])
    else:
        inf_losses_data = np.load(inf_losses_path)
        inf_train_losses = inf_losses_data["train_losses"]
        inf_test_losses = inf_losses_data["test_losses"]
        inf_fine_tune_train_losses = inf_losses_data["fine_tune_train_losses"]
        inf_fine_tune_test_losses = inf_losses_data["fine_tune_test_losses"]
    inf_epochs = np.arange(1, len(inf_train_losses) + 1) if len(inf_train_losses) > 0 else np.array([])
    inf_fine_tune_epochs = np.arange(len(inf_train_losses) + 1, len(inf_train_losses) + len(inf_fine_tune_train_losses) + 1) if len(inf_fine_tune_train_losses) > 0 else np.array([])

    # Create subplots (2x3 layout)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # VAE losses
    if len(vae_train_losses) > 0:
        axes[0, 0].plot(vae_epochs, vae_train_losses, label="train", linewidth=2)
        axes[0, 0].plot(vae_epochs, vae_test_losses, label="test", linewidth=2, linestyle="--")
        axes[0, 0].set_yscale("log")
    else:
        axes[0, 0].text(0.5, 0.5, "No VAE data", ha="center", va="center", transform=axes[0, 0].transAxes)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("VAE Loss Curves")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Generator losses (main training)
    if len(gen_train_losses) > 0:
        axes[0, 1].plot(gen_epochs, gen_train_losses, label="train", linewidth=2, color="orange")
        axes[0, 1].plot(gen_epochs, gen_test_losses, label="test", linewidth=2, linestyle="--", color="red")
        axes[0, 1].set_yscale("log")
    else:
        axes[0, 1].text(0.5, 0.5, "No Generator data", ha="center", va="center", transform=axes[0, 1].transAxes)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Generator Loss Curves (Main Training)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Generator fine-tune losses
    if len(gen_fine_tune_train_losses) > 0:
        axes[0, 2].plot(gen_fine_tune_epochs, gen_fine_tune_train_losses, label="fine-tune train", linewidth=2, color="green")
        axes[0, 2].plot(gen_fine_tune_epochs, gen_fine_tune_test_losses, label="fine-tune test", linewidth=2, linestyle="--", color="darkgreen")
        axes[0, 2].set_yscale("log")
    else:
        axes[0, 2].text(0.5, 0.5, "No Fine-tune data", ha="center", va="center", transform=axes[0, 2].transAxes)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Loss")
    axes[0, 2].set_title("Generator Loss Curves (Fine-tuning)")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Inferrer losses (main training)
    if len(inf_train_losses) > 0:
        axes[1, 0].plot(inf_epochs, inf_train_losses, label="train", linewidth=2, color="purple")
        axes[1, 0].plot(inf_epochs, inf_test_losses, label="test", linewidth=2, linestyle="--", color="magenta")
        axes[1, 0].set_yscale("log")
    else:
        axes[1, 0].text(0.5, 0.5, "No Inferrer data", ha="center", va="center", transform=axes[1, 0].transAxes)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Inferrer Loss Curves (Main Training)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Inferrer fine-tune losses
    if len(inf_fine_tune_train_losses) > 0:
        axes[1, 1].plot(inf_fine_tune_epochs, inf_fine_tune_train_losses, label="fine-tune train", linewidth=2, color="cyan")
        axes[1, 1].plot(inf_fine_tune_epochs, inf_fine_tune_test_losses, label="fine-tune test", linewidth=2, linestyle="--", color="darkcyan")
        axes[1, 1].set_yscale("log")
    else:
        axes[1, 1].text(0.5, 0.5, "No Fine-tune data", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Inferrer Loss Curves (Fine-tuning)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Empty subplot for symmetry
    axes[1, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(folder, f"{label}_all_loss_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[plot_loss_curves] figure saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def visualize_param_in_latent_space(
    model_path: str,
    folder: str,
    label: str,
    latent_dim: int = 3,
    max_samples: int = 1000,
    save_path: str = None,
):
    """
    Visualize the distribution of the parameters from the dataset in the latent space (assume latent_dim=3).
    Plots both mu and logvar distributions colored by all parameters (up to 3).
    Args:
        model_path: Path to saved model state dict
        folder: Path to folder containing training data
        label: Label for the dataset
        latent_dim: Dimensionality of the latent space (should be 3)
        max_samples: Maximum number of samples to use for visualization
        save_path: Optional path to save the visualization
    """
    assert latent_dim == 3, "This function assumes latent_dim=3 for 3D scatter plot."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load trained model
    model = VAE(latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # Load training data
    train_loader, _ = create_dataloader(folder, label, "train", batch_size=32, shuffle=False)
    latent_mus = []
    latent_logvars = []
    params = []
    with torch.no_grad():
        sample_count = 0
        for x, _, p in train_loader:
            if sample_count >= max_samples:
                break
            x = x.to(device)
            mu, logvar = model.encoder(x)
            latent_mus.append(mu.cpu().numpy())
            latent_logvars.append(logvar.cpu().numpy())
            params.append(p.cpu().numpy())
            sample_count += x.size(0)
    latent_mus = np.concatenate(latent_mus, axis=0)[:max_samples]
    latent_logvars = np.concatenate(latent_logvars, axis=0)[:max_samples]
    params = np.concatenate(params, axis=0)[:max_samples]
    # Denormalize parameters for color mapping
    param1 = []
    param2 = []
    param3 = []
    for i in range(params.shape[0]):
        _, p_denorm = denormalize_generated_Iq(folder, label, np.zeros(100), params[i])
        param1.append(p_denorm[0])
        if len(p_denorm) > 1:
            param2.append(p_denorm[1])
        if len(p_denorm) > 2:
            param3.append(p_denorm[2])
    param1 = np.array(param1)
    param2 = np.array(param2) if len(param2) > 0 else None
    param3 = np.array(param3) if len(param3) > 0 else None
    # 3D scatter plots: mu/param1, mu/param2, mu/param3, logvar/param1, logvar/param2, logvar/param3
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12, 8))
    # mu colored by param1
    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    sc1 = ax1.scatter(
        latent_mus[:, 0],
        latent_mus[:, 1],
        latent_mus[:, 2],
        c=param1,
        cmap="viridis",
        s=20,
        alpha=0.7,
        label="Samples"
    )
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label("Parameter 1 (denormalized)")
    ax1.set_xlabel("Latent mu 1")
    ax1.set_ylabel("Latent mu 2")
    ax1.set_zlabel("Latent mu 3")
    ax1.set_title("Parameter 1 in latent mu space")
    # mu colored by param2
    if param2 is not None and param2.shape[0] == latent_mus.shape[0]:
        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        sc2 = ax2.scatter(
            latent_mus[:, 0],
            latent_mus[:, 1],
            latent_mus[:, 2],
            c=param2,
            cmap="plasma",
            s=20,
            alpha=0.7,
            label="Samples"
        )
        cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
        cbar2.set_label("Parameter 2 (denormalized)")
        ax2.set_xlabel("Latent mu 1")
        ax2.set_ylabel("Latent mu 2")
        ax2.set_zlabel("Latent mu 3")
        ax2.set_title("Parameter 2 in latent mu space")
    else:
        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        ax2.set_title("No Parameter 2")
    # mu colored by param3
    if param3 is not None and param3.shape[0] == latent_mus.shape[0]:
        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        sc3 = ax3.scatter(
            latent_mus[:, 0],
            latent_mus[:, 1],
            latent_mus[:, 2],
            c=param3,
            cmap="inferno",
            s=20,
            alpha=0.7,
            label="Samples"
        )
        cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.1)
        cbar3.set_label("Parameter 3 (denormalized)")
        ax3.set_xlabel("Latent mu 1")
        ax3.set_ylabel("Latent mu 2")
        ax3.set_zlabel("Latent mu 3")
        ax3.set_title("Parameter 3 in latent mu space")
    else:
        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        ax3.set_title("No Parameter 3")
    # logvar colored by param1
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    sc4 = ax4.scatter(
        latent_logvars[:, 0],
        latent_logvars[:, 1],
        latent_logvars[:, 2],
        c=param1,
        cmap="viridis",
        s=20,
        alpha=0.7,
        label="Samples"
    )
    cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.1)
    cbar4.set_label("Parameter 1 (denormalized)")
    ax4.set_xlabel("Latent logvar 1")
    ax4.set_ylabel("Latent logvar 2")
    ax4.set_zlabel("Latent logvar 3")
    ax4.set_title("Parameter 1 in latent logvar space")
    # logvar colored by param2
    if param2 is not None and param2.shape[0] == latent_logvars.shape[0]:
        ax5 = fig.add_subplot(2, 3, 5, projection="3d")
        sc5 = ax5.scatter(
            latent_logvars[:, 0],
            latent_logvars[:, 1],
            latent_logvars[:, 2],
            c=param2,
            cmap="plasma",
            s=20,
            alpha=0.7,
            label="Samples"
        )
        cbar5 = plt.colorbar(sc5, ax=ax5, pad=0.1)
        cbar5.set_label("Parameter 2 (denormalized)")
        ax5.set_xlabel("Latent logvar 1")
        ax5.set_ylabel("Latent logvar 2")
        ax5.set_zlabel("Latent logvar 3")
        ax5.set_title("Parameter 2 in latent logvar space")
    else:
        ax5 = fig.add_subplot(2, 3, 5, projection="3d")
        ax5.set_title("No Parameter 2")
    # logvar colored by param3
    if param3 is not None and param3.shape[0] == latent_logvars.shape[0]:
        ax6 = fig.add_subplot(2, 3, 6, projection="3d")
        sc6 = ax6.scatter(
            latent_logvars[:, 0],
            latent_logvars[:, 1],
            latent_logvars[:, 2],
            c=param3,
            cmap="inferno",
            s=20,
            alpha=0.7,
            label="Samples"
        )
        cbar6 = plt.colorbar(sc6, ax=ax6, pad=0.1)
        cbar6.set_label("Parameter 3 (denormalized)")
        ax6.set_xlabel("Latent logvar 1")
        ax6.set_ylabel("Latent logvar 2")
        ax6.set_zlabel("Latent logvar 3")
        ax6.set_title("Parameter 3 in latent logvar space")
    else:
        ax6 = fig.add_subplot(2, 3, 6, projection="3d")
        ax6.set_title("No Parameter 3")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    plt.show()
    return latent_mus, latent_logvars, param1, param2, param3

def show_vae_random_reconstructions(
    folder: str,
    label: str,
    model_path: str | None = None,
    model: VAE | None = None,
    latent_dim: int = 3,
    num_samples: int = 4,
    device: str | torch.device | None = None,
):
    """
    Show random reconstructions from the VAE model.
    Args:
        folder: Path to data folder
        label: Dataset label
        model_path: Path to saved model (if model not provided)
        model: Pre-loaded VAE model (if model_path not provided)
        latent_dim: Latent dimension
        num_samples: Number of samples to show
        device: Device to use
    """
    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    # Load/validate the model
    if model is None:
        assert model_path is not None, "Provide `model` or `model_path`."
        model = VAE(latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = model.to(device)
    model.eval()
    # Make a fresh DataLoader with shuffle=True so we can draw random items
    loader, _ = create_dataloader(folder, label, "train", batch_size=1, shuffle=True)
    figs = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Grab one random sample
            x, q, p = next(iter(loader))
            x = x.to(device)
            # Get reconstruction using deterministic forward pass (use mu directly)
            # mu, logvar = model.encoder(x)
            # recon = model.decoder(mu.unsqueeze(0)).squeeze(0)  # Use mu directly for deterministic reconstruction
            recon_avg, mu, logvar = model(x)  # Use the mean reconstruction from the ensemble
            # Detach to CPU and numpy
            x_np = x.squeeze().cpu().numpy()  # (100,)
            p_np = p.squeeze().cpu().numpy()  # (input_dim,)
            recon_np = recon_avg.squeeze().cpu().numpy()  # (100,)
            mu_np = mu.squeeze().cpu().numpy()  # (latent_dim,)
            figs.append((x_np, recon_np, mu_np, p_np))
    # Plotting
    n_rows = num_samples
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)
    for idx, (x_np, recon_np, mu_np, p_np) in enumerate(figs):
        # Input 1D signal
        ax_in = axes[idx, 0]
        x_np, p_np = denormalize_generated_Iq(folder, label, x_np, p_np)  # Assuming you have a function to denormalize
        ax_in.plot(x_np, "b-", linewidth=1.5, label="Input")
        ax_in.set_title(f"Input #{idx}")
        ax_in.set_xlabel("Feature index")
        ax_in.set_ylabel("log10(I(q))")
        ax_in.grid(True, alpha=0.3)
        ax_in.legend()
        # Reconstruction
        ax_out = axes[idx, 1]
        recon_np, _ = denormalize_generated_Iq(folder, label, recon_np, p_np)  # Assuming you have a function to denormalize
        ax_out.plot(recon_np, "r-", linewidth=1.5, label="Reconstruction")
        mu_str = ", ".join([f"{v:+.3f}" for v in mu_np])
        ax_out.set_title(f"Recon #{idx} params = [{p_np[0]:+.3f}, {p_np[1]:+.3f}]\nLatent mu = [{mu_str}]")
        ax_out.set_xlabel("Feature index")
        ax_out.set_ylabel("log10(I(q))")
        ax_out.grid(True, alpha=0.3)
        ax_out.legend()
        # Difference
        ax_diff = axes[idx, 2]
        diff_np = x_np - recon_np
        ax_diff.plot(diff_np, "g-", linewidth=1.5, label="Difference")
        rmse = np.sqrt(np.mean(diff_np**2))
        ax_diff.set_title(f"Difference #{idx}\nRMSE = {rmse:.6f}")
        ax_diff.set_xlabel("Feature index")
        ax_diff.set_ylabel("Difference")
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/{label}_random_reconstructions.png", dpi=300)
    plt.show()


def show_gen_random_reconstruction(
    folder: str,
    label: str,
    model_path: str | None = None,
    model: Generator | None = None,
    latent_dim: int = 3,
    input_dim: int = 3,
    num_samples: int = 4,
    device: str | torch.device | None = None,
):
    """
    Show random reconstructions from the Generator model.
    Args:
        folder: Path to data folder
        label: Dataset label
        model_path: Path to saved model (if model not provided)
        model: Pre-loaded Generator model (if model_path not provided)
        latent_dim: Latent dimension
        input_dim: Input parameter dimension
        num_samples: Number of samples to show
        device: Device to use
    """
    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    # Load/validate the model
    if model is None:
        assert model_path is not None, "Provide `model` or `model_path`."
        model = Generator(input_dim=input_dim, latent_dim=latent_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = model.to(device)
    model.eval()
    # Make a fresh DataLoader with shuffle=True so we can draw random items
    loader, _ = create_dataloader(folder, label, "train", batch_size=1, shuffle=True)
    figs = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Grab one random sample
            log10Iq, q, p = next(iter(loader))
            log10Iq, p = log10Iq.to(device), p.to(device)
            # Get reconstruction using deterministic forward pass (use mu directly)
            # mu, logvar = model.cvtp2l(p)
            # recon = model.decoder(mu.unsqueeze(0)).squeeze(0)  # Use mu directly for deterministic reconstruction
            recon_avg, mu, logvar = model(p)  # Use the mean reconstruction from the ensemble
            # Detach to CPU and numpy
            Iq_np = log10Iq.squeeze().cpu().numpy()  # (100,)
            recon_np = recon_avg.squeeze().cpu().numpy()  # (100,)
            p_np = p.squeeze().cpu().numpy()  # (input_dim,)
            figs.append((Iq_np, recon_np, p_np))
    # Plotting
    n_rows = num_samples
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 3)
    for idx, (Iq_np, recon_np, p_np) in enumerate(figs):
        # Input 1D signal
        ax_in = axes[idx, 0]
        Iq_np, p_np = denormalize_generated_Iq(folder, label, Iq_np, p_np)
        ax_in.plot(Iq_np, "b-", linewidth=1.5, label="Input")
        ax_in.set_title(f"Input #{idx}, params = [{p_np[0]:+.3f}, {p_np[1]:+.3f}, {p_np[2]:+.3f}]")
        ax_in.set_xlabel("Feature index")
        ax_in.set_ylabel("log10(I(q))")
        ax_in.grid(True, alpha=0.3)
        ax_in.legend()
        # Reconstruction
        ax_out = axes[idx, 1]
        recon_np, _ = denormalize_generated_Iq(folder, label, recon_np, p_np)
        ax_out.plot(recon_np, "r-", linewidth=1.5, label="Reconstruction")
        p_str = ", ".join([f"{v:+.3f}" for v in p_np])
        ax_out.set_title(f"Gen #{idx}\nParams = [{p_str}]")
        ax_out.set_xlabel("Feature index")
        ax_out.set_ylabel("log10(I(q))")
        ax_out.grid(True, alpha=0.3)
        ax_out.legend()
        # Difference
        ax_diff = axes[idx, 2]
        diff_np = Iq_np - recon_np
        ax_diff.plot(diff_np, "g-", linewidth=1.5, label="Difference")
        rmse = np.sqrt(np.mean(diff_np**2))
        ax_diff.set_title(f"Difference #{idx}\nRMSE = {rmse:.6f}")
        ax_diff.set_xlabel("Feature index")
        ax_diff.set_ylabel("Difference")
        ax_diff.grid(True, alpha=0.3)
        ax_diff.legend()
    plt.tight_layout()
    plt.savefig(f"{folder}/{label}_gen_random_reconstructions.png", dpi=300)
    plt.show()


def show_infer_random_analysis(
    folder: str,
    label: str,
    model_path: str | None = None,
    model: Inferrer | None = None,
    input_dim: int = 100,
    latent_dim: int = 3,
    output_dim: int = 2,
    device: str | torch.device | None = None,
):
    """
    Show random analysis from the Inferrer model.
    Args:
        folder: Path to data folder
        label: Dataset label
        model_path: Path to saved model (if model not provided)
        model: Pre-loaded Inferrer model (if model_path not provided)
        input_dim: Input dimension
        latent_dim: Latent dimension
        output_dim: Output dimension
        device: Device to use
    """
    # Resolve device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    # Load/validate the model
    if model is None:
        assert model_path is not None, "Provide `model` or `model_path`."
        model = Inferrer(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = model.to(device)
    model.eval()
    # Evaluate on entire test set
    test_loader, _ = create_dataloader(folder, label, "test", batch_size=32, shuffle=False)

    all_true_params = []
    all_pred_params = []

    with torch.no_grad():
        for log10Iq, q, p in test_loader:
            log10Iq, p = log10Iq.to(device), p.to(device)
            # Get predictions from inferrer
            pred_avg, mu, logvar = model(log10Iq)

            # Collect true and predicted parameters
            all_true_params.append(p.cpu().numpy())
            all_pred_params.append(pred_avg.cpu().numpy())

    # Concatenate all batches
    all_true_params = np.concatenate(all_true_params, axis=0)
    all_pred_params = np.concatenate(all_pred_params, axis=0)

    # Denormalize parameters for plotting
    all_true_params_denorm = []
    all_pred_params_denorm = []

    for i in range(len(all_true_params)):
        _, true_denorm = denormalize_generated_Iq(folder, label, np.zeros(100), all_true_params[i])
        _, pred_denorm = denormalize_generated_Iq(folder, label, np.zeros(100), all_pred_params[i])
        all_true_params_denorm.append(true_denorm)
        all_pred_params_denorm.append(pred_denorm)

    all_true_params_denorm = np.array(all_true_params_denorm)
    all_pred_params_denorm = np.array(all_pred_params_denorm)

    # Create scatter plots
    fig, axes = plt.subplots(1, output_dim, figsize=(6*output_dim, 5))
    if output_dim == 1:
        axes = [axes]

    param_names = ['Parameter 1', 'Parameter 2'] if output_dim == 2 else [f'Parameter {i+1}' for i in range(output_dim)]

    for i in range(output_dim):
        ax = axes[i]

        # Scatter plot
        ax.scatter(all_true_params_denorm[:, i], all_pred_params_denorm[:, i],
                  alpha=0.6, s=20, edgecolors='none')

        # Perfect prediction line
        min_val = min(all_true_params_denorm[:, i].min(), all_pred_params_denorm[:, i].min())
        max_val = max(all_true_params_denorm[:, i].max(), all_pred_params_denorm[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        # Calculate R² and RMSE
        r2 = np.corrcoef(all_true_params_denorm[:, i], all_pred_params_denorm[:, i])[0, 1]**2
        rmse = np.sqrt(np.mean((all_true_params_denorm[:, i] - all_pred_params_denorm[:, i])**2))

        ax.set_xlabel(f'True {param_names[i]}')
        ax.set_ylabel(f'Predicted {param_names[i]}')
        ax.set_title(f'{param_names[i]}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{folder}/{label}_inferrer_test_predictions.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Evaluated {len(all_true_params)} test samples")
    print(f"Overall prediction statistics:")
    for i in range(output_dim):
        r2 = np.corrcoef(all_true_params_denorm[:, i], all_pred_params_denorm[:, i])[0, 1]**2
        rmse = np.sqrt(np.mean((all_true_params_denorm[:, i] - all_pred_params_denorm[:, i])**2))
        print(f"  {param_names[i]}: R² = {r2:.4f}, RMSE = {rmse:.4f}")


def LS_fit_params_with_gen(target_log10Iq, gen_model=None, model_path=None, folder=None, label=None, latent_dim=3, input_dim=3, target_loss=1e-6, max_steps=3000, lr=1e-2):
    """
    Fit parameters using least squares to match the target_log10Iq scattering function using the generator model.

    Args:
        target_log10Iq: The target log10(I(q)) array to fit (numpy array of shape (100,))
        gen_model: The trained Generator model (optional if model_path provided)
        model_path: Path to the saved generator model state dict (optional if gen_model provided)
        folder: Path to the data folder
        label: Dataset label for loading stats
        latent_dim: Latent dimension (default 3)
        input_dim: Input parameter dimension (default 3)
        target_loss: Target loss threshold to stop optimization (default 1e-6)
        max_steps: Maximum number of optimization steps (default 1000)
        lr: Learning rate for optimization (default 1e-2)

    Returns:
        fitted_params: The fitted parameters (denormalized)
        final_loss: The final loss value
        param_history: List of parameter values at each step (normalized)
        loss_history: List of loss values at each step
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load/validate the model
    if gen_model is None:
        assert model_path is not None, "Provide `gen_model` or `model_path`."
        gen_model = Generator(input_dim=input_dim, latent_dim=latent_dim).to(device)
        gen_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        gen_model = gen_model.to(device)
    gen_model.eval()    # Load normalization stats
    data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    log10Iq_mean = data_stats["mean_log10Iq"]
    log10Iq_std = data_stats["std_log10Iq"]
    params_name = data_stats["params_name"]
    params_mean = data_stats["mean_params"]
    params_std = data_stats["std_params"]
    min_params = data_stats["min_params"]
    max_params = data_stats["max_params"]

    # Remove sigmaL if present (as done in dataset)
    if "sigmaL" in params_name:
        sigmaL_index = list(params_name).index("sigmaL")
        params_mean = np.delete(params_mean, sigmaL_index, axis=0)
        params_std = np.delete(params_std, sigmaL_index, axis=0)
        min_params = np.delete(min_params, sigmaL_index, axis=0)
        max_params = np.delete(max_params, sigmaL_index, axis=0)
        params_name = np.delete(params_name, sigmaL_index)

    # Normalize parameter bounds for optimization
    min_params_norm = (min_params - params_mean) / params_std
    max_params_norm = (max_params - params_mean) / params_std

    # Normalize target Iq
    target_norm = (target_log10Iq - log10Iq_mean) / log10Iq_std
    target_norm = torch.tensor(target_norm, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 100)

    # Initialize parameters to optimize (start from random normal, since normalized)
    params_init = torch.randn(input_dim, requires_grad=True, device=device)

    # Optimizer
    optimizer = optim.Adam([params_init], lr=lr)

    # Track convergence
    param_history = []
    loss_history = []

    print(f"Starting least squares fitting with target_loss={target_loss}, max_steps={max_steps}, lr={lr}")

    for step in range(max_steps):
        optimizer.zero_grad()

        # Generate Iq from current parameters
        gen_Iq, _, _ = gen_model(params_init.unsqueeze(0))  # (1, 100)

        # Compute MSE loss
        loss = F.mse_loss(gen_Iq, target_norm)

        # Backprop and update
        loss.backward()
        optimizer.step()

        # Clamp parameters to normalized bounds
        with torch.no_grad():
            params_init.data.clamp_(min_params_norm, max_params_norm)

        # Track history
        param_history.append(params_init.detach().cpu().numpy().copy())
        loss_history.append(loss.item())

        # Check convergence
        if loss.item() < target_loss:
            print(f"Converged at step {step+1} with loss {loss.item():.2e}")
            break

        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{max_steps}, Loss: {loss.item():.2e}")

    # Denormalize fitted parameters
    params_norm = params_init.detach().cpu().numpy()
    fitted_params = params_norm * params_std + params_mean

    final_loss = loss.item()
    print(f"Fitting completed. Final loss: {final_loss:.2e}")
    print(f"Fitted parameters: {fitted_params}")

    return fitted_params, final_loss, param_history, loss_history


def show_sample_LS_fitting(folder, label, model_path, num_samples=3, num_fits_per_sample=3, latent_dim=3, input_dim=3, target_loss=1e-6, max_steps=1000, lr=1e-2):
    """
    Plot least squares fitting results for sample log10Iq data.

    For each of num_samples test samples, performs num_fits_per_sample fittings with different
    random initializations to show convergence. Plots Iq comparisons and parameter trajectories.

    Args:
        folder: Path to data folder
        label: Dataset label
        model_path: Path to saved generator model
        num_samples: Number of test samples to use (default 3)
        num_fits_per_sample: Number of fits per sample (default 3)
        latent_dim: Latent dimension (default 3)
        input_dim: Input parameter dimension (default 3)
        target_loss: Target loss for fitting (default 1e-6)
        max_steps: Max steps for fitting (default 1000)
        lr: Learning rate for fitting (default 1e-2)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Load test data
    test_loader, _ = create_dataloader(folder, label, "test", batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    gen_model = Generator(input_dim=input_dim, latent_dim=latent_dim).to(device)
    gen_model.load_state_dict(torch.load(model_path, map_location=device))
    gen_model.eval()

    # Get data stats for denormalization
    data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    log10Iq_mean = data_stats["mean_log10Iq"]
    log10Iq_std = data_stats["std_log10Iq"]
    params_name = data_stats["params_name"]
    params_mean = data_stats["mean_params"]
    params_std = data_stats["std_params"]

    if "sigmaL" in params_name:
        sigmaL_index = list(params_name).index("sigmaL")
        params_mean = np.delete(params_mean, sigmaL_index, axis=0)
        params_std = np.delete(params_std, sigmaL_index, axis=0)
        params_name = np.delete(params_name, sigmaL_index)

    # Select num_samples random samples
    samples = []
    for i in range(num_samples):
        log10Iq, q, p = next(iter(test_loader))
        log10Iq = log10Iq.squeeze().numpy()
        p = p.squeeze().numpy()
        # Denormalize
        log10Iq_denorm, p_denorm = denormalize_generated_Iq(folder, label, log10Iq, p)
        samples.append((log10Iq_denorm, p_denorm))

    # For each sample, fit num_fits_per_sample times
    all_fitted_params = []
    all_param_histories = []
    all_loss_histories = []

    for sample_idx, (target_Iq, true_p) in enumerate(samples):
        fitted_for_sample = []
        histories_for_sample = []
        losses_for_sample = []

        print(f"Fitting sample {sample_idx+1}/{num_samples}")
        for fit_idx in range(num_fits_per_sample):
            fitted_p, final_loss, param_hist, loss_hist = LS_fit_params_with_gen(
                target_Iq, gen_model=gen_model, folder=folder, label=label,
                latent_dim=latent_dim, input_dim=input_dim,
                target_loss=target_loss, max_steps=max_steps, lr=lr
            )
            fitted_for_sample.append(fitted_p)
            histories_for_sample.append(param_hist)
            losses_for_sample.append(loss_hist)

        all_fitted_params.append(fitted_for_sample)
        all_param_histories.append(histories_for_sample)
        all_loss_histories.append(losses_for_sample)

    # Now plot
    fig = plt.figure(figsize=(18, 6 * num_samples))

    for sample_idx in range(num_samples):
        # Iq comparison
        ax1 = fig.add_subplot(num_samples, 2, 2 * sample_idx + 1)
        target_Iq = samples[sample_idx][0]
        ax1.plot(target_Iq, 'b-', label='Target Iq', linewidth=2)

        for fit_idx in range(num_fits_per_sample):
            # Generate fitted Iq
            fitted_p_norm = (all_fitted_params[sample_idx][fit_idx] - params_mean) / params_std
            fitted_p_tensor = torch.tensor(fitted_p_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                gen_Iq, _, _ = gen_model(fitted_p_tensor)
            gen_Iq_denorm, _ = denormalize_generated_Iq(folder, label, gen_Iq.squeeze().cpu().numpy(), fitted_p_norm)
            ax1.plot(gen_Iq_denorm, '--', label=f'Fitted Iq {fit_idx+1}', linewidth=1.5)

        ax1.set_xlabel('q index')
        ax1.set_ylabel('log10(I(q))')
        ax1.set_title(f'Sample {sample_idx+1}: Iq Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 3D parameter trajectory
        ax2 = fig.add_subplot(num_samples, 2, 2 * sample_idx + 2, projection='3d')
        for fit_idx in range(num_fits_per_sample):
            param_hist = np.array(all_param_histories[sample_idx][fit_idx])
            ax2.plot(param_hist[:, 0], param_hist[:, 1], param_hist[:, 2], label=f'Fit {fit_idx+1}')
            ax2.scatter(param_hist[0, 0], param_hist[0, 1], param_hist[0, 2], marker='o', s=50, label=f'Start {fit_idx+1}')
            ax2.scatter(param_hist[-1, 0], param_hist[-1, 1], param_hist[-1, 2], marker='x', s=50, label=f'End {fit_idx+1}')

        # Plot true params
        true_p_norm = (samples[sample_idx][1] - params_mean) / params_std
        ax2.scatter(true_p_norm[0], true_p_norm[1], true_p_norm[2], marker='*', s=100, c='red', label='True params')

        ax2.set_xlabel('Param 1 (norm)')
        ax2.set_ylabel('Param 2 (norm)')
        ax2.set_zlabel('Param 3 (norm)')
        ax2.set_title(f'Sample {sample_idx+1}: Parameter Trajectories')
        ax2.legend()

    plt.tight_layout()
    save_path = f"{folder}/{label}_LS_fitting_samples.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()


def fit_test_data(folder: str, label: str, model_path: str, latent_dim: int = 3, input_dim: int = 3, target_loss: float = 1e-6, max_steps: int = 3000, lr: float = 1e-2):
    """
    Fit parameters for all test data using least squares with the trained generator model.

    Args:
        folder: Path to the data folder
        label: Dataset label
        model_path: Path to the saved generator model state dict
        latent_dim: Latent dimension (default 3)
        input_dim: Input parameter dimension (default 3)
        target_loss: Target loss threshold for fitting (default 1e-6)
        max_steps: Maximum number of optimization steps (default 1000)
        lr: Learning rate for optimization (default 1e-2)

    Saves fitted and true parameters to {folder}/{label}_test_fits.npz
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_loader, _ = create_dataloader(folder, label, "test", batch_size=1, shuffle=False)
    print(f"Loaded {len(test_loader.dataset)} test samples")

    # Load generator model
    gen_model = Generator(input_dim=input_dim, latent_dim=latent_dim).to(device)
    gen_model.load_state_dict(torch.load(model_path, map_location=device))
    gen_model.eval()

    # Get data stats for denormalization
    data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    params_name = data_stats["params_name"]
    params_mean = data_stats["mean_params"]
    params_std = data_stats["std_params"]

    if "sigmaL" in params_name:
        sigmaL_index = list(params_name).index("sigmaL")
        params_mean = np.delete(params_mean, sigmaL_index, axis=0)
        params_std = np.delete(params_std, sigmaL_index, axis=0)
        params_name = np.delete(params_name, sigmaL_index)

    # Store results
    all_true_params = []
    all_fitted_params = []
    all_final_losses = []

    print("Starting least squares fitting for all test data...")

    for idx, (log10Iq, q, p) in enumerate(test_loader):
        if idx % 50 == 0:
            print(f"Processing sample {idx+1}/{len(test_loader.dataset)}")

        # Denormalize target Iq
        log10Iq_np = log10Iq.squeeze().numpy()
        p_np = p.squeeze().numpy()
        target_Iq_denorm, true_p_denorm = denormalize_generated_Iq(folder, label, log10Iq_np, p_np)

        # Fit parameters
        fitted_p, final_loss, _, _ = LS_fit_params_with_gen(
            target_Iq_denorm, gen_model=gen_model, folder=folder, label=label,
            latent_dim=latent_dim, input_dim=input_dim,
            target_loss=target_loss, max_steps=max_steps, lr=lr
        )

        all_true_params.append(true_p_denorm)
        all_fitted_params.append(fitted_p)
        all_final_losses.append(final_loss)

    # Convert to numpy arrays
    all_true_params = np.array(all_true_params)
    all_fitted_params = np.array(all_fitted_params)
    all_final_losses = np.array(all_final_losses)

    # Save to npz file
    save_path = os.path.join(folder, f"{label}_test_fits.npz")
    np.savez(save_path,
             true_params=all_true_params,
             fitted_params=all_fitted_params,
             final_losses=all_final_losses,
             params_name=params_name)

    print(f"Saved fitted results to {save_path}")
    print(f"Mean final loss: {all_final_losses.mean():.2e}")
    print(f"Std final loss: {all_final_losses.std():.2e}")

    return all_true_params, all_fitted_params, all_final_losses


def visualize_LS_fitting_performance(fit_test_data_path):
    """
    Visualize the least squares fitting performance from the fit_test_data.npz file.

    Args:
        fit_test_data_path: Path to the .npz file containing fitting results
    """
    # Load the fitting data
    data = np.load(fit_test_data_path)
    true_params = data['true_params']
    fitted_params = data['fitted_params']
    final_losses = data['final_losses']
    params_name = data['params_name']

    # Remove sigmaL if present (as done elsewhere)
    if "sigmaL" in params_name:
        sigmaL_index = list(params_name).index("sigmaL")
        params_name = np.delete(params_name, sigmaL_index)

    num_params = true_params.shape[1]

    # Create subplots: one for each parameter scatter plot, plus one for loss histogram
    fig, axes = plt.subplots(1, num_params + 1, figsize=(6*(num_params + 1), 5))

    param_names = [f'Parameter {i+1}' for i in range(num_params)]

    # Scatter plots for each parameter
    for i in range(num_params):
        ax = axes[i]

        # Scatter plot
        ax.scatter(true_params[:, i], fitted_params[:, i], alpha=0.6, s=20, edgecolors='none')

        # Perfect prediction line
        min_val = min(true_params[:, i].min(), fitted_params[:, i].min())
        max_val = max(true_params[:, i].max(), fitted_params[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')

        # Calculate R² and RMSE
        r2 = np.corrcoef(true_params[:, i], fitted_params[:, i])[0, 1]**2
        rmse = np.sqrt(np.mean((true_params[:, i] - fitted_params[:, i])**2))

        ax.set_xlabel(f'True {param_names[i]}')
        ax.set_ylabel(f'Fitted {param_names[i]}')
        ax.set_title(f'{param_names[i]}\nR² = {r2:.4f}, RMSE = {rmse:.4f}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')

    # Loss histogram
    ax_loss = axes[-1]
    ax_loss.hist(final_losses, bins=50, alpha=0.7, edgecolor='black')
    ax_loss.set_xlabel('Final Loss')
    ax_loss.set_ylabel('Frequency')
    ax_loss.set_title(f'Final Loss Distribution\nMean: {final_losses.mean():.2e}, Std: {final_losses.std():.2e}')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_yscale('log')  # Since losses can span orders of magnitude

    plt.tight_layout()

    # Save the plot
    save_path = fit_test_data_path.replace('.npz', '_fitting_performance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fitting performance visualization saved to {save_path}")
    plt.show()

    # Print summary statistics
    print("\nFitting Performance Summary:")
    print(f"Total samples: {len(final_losses)}")
    print(f"Mean final loss: {final_losses.mean():.2e}")
    print(f"Std final loss: {final_losses.std():.2e}")
    print(f"Min final loss: {final_losses.min():.2e}")
    print(f"Max final loss: {final_losses.max():.2e}")
    print(f"Median final loss: {np.median(final_losses):.2e}")

    for i in range(num_params):
        r2 = np.corrcoef(true_params[:, i], fitted_params[:, i])[0, 1]**2
        rmse = np.sqrt(np.mean((true_params[:, i] - fitted_params[:, i])**2))
        mae = np.mean(np.abs(true_params[:, i] - fitted_params[:, i]))
        print(f"\n{param_names[i]}:")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
