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
    log10Iq_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    log10Iq_mean = log10Iq_stats["mean"]
    log10Iq_std = log10Iq_stats["std"]
    params_mean = log10Iq_stats["params_mean"]
    params_std = log10Iq_stats["params_std"]
    log10Iq = log10Iq_norm * log10Iq_std + log10Iq_mean  # denormalize)
    params = params_norm * params_std + params_mean  # denormalize parameters
    return log10Iq, params


def normalize_Iq(folder, label, log10Iq, params):
    data_stats = np.load(f"{folder}/{label}_log10Iq_dataset_train_stats.npz")
    log10Iq_mean = data_stats["mean_log10Iq"]
    log10Iq_std = data_stats["std_log10Iq"]
    params_mean = data_stats["mean_params"]
    params_std = data_stats["std_params"]
    log10Iq_norm = (log10Iq - log10Iq_mean) / log10Iq_std  # normalize to mean=0, std=1 for each q point
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
        epsilons = torch.randn(100, *mu.shape, device=mu.device)
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
    input_dim: int = 2,
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
    Plots both mu and logvar distributions colored by parameter 1 and 2.
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
    for i in range(params.shape[0]):
        _, p_denorm = denormalize_generated_Iq(folder, label, np.zeros(100), params[i])
        param1.append(p_denorm[0])
        if len(p_denorm) > 1:
            param2.append(p_denorm[1])
    param1 = np.array(param1)
    param2 = np.array(param2) if len(param2) > 0 else None
    # 3D scatter plots: mu/param1, mu/param2, logvar/param1, logvar/param2
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 8))
    # mu colored by param1
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
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
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
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
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.set_title("No Parameter 2")
    # logvar colored by param1
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    sc3 = ax3.scatter(
        latent_logvars[:, 0],
        latent_logvars[:, 1],
        latent_logvars[:, 2],
        c=param1,
        cmap="viridis",
        s=20,
        alpha=0.7,
        label="Samples"
    )
    cbar3 = plt.colorbar(sc3, ax=ax3, pad=0.1)
    cbar3.set_label("Parameter 1 (denormalized)")
    ax3.set_xlabel("Latent logvar 1")
    ax3.set_ylabel("Latent logvar 2")
    ax3.set_zlabel("Latent logvar 3")
    ax3.set_title("Parameter 1 in latent logvar space")
    # logvar colored by param2
    if param2 is not None and param2.shape[0] == latent_logvars.shape[0]:
        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        sc4 = ax4.scatter(
            latent_logvars[:, 0],
            latent_logvars[:, 1],
            latent_logvars[:, 2],
            c=param2,
            cmap="plasma",
            s=20,
            alpha=0.7,
            label="Samples"
        )
        cbar4 = plt.colorbar(sc4, ax=ax4, pad=0.1)
        cbar4.set_label("Parameter 2 (denormalized)")
        ax4.set_xlabel("Latent logvar 1")
        ax4.set_ylabel("Latent logvar 2")
        ax4.set_zlabel("Latent logvar 3")
        ax4.set_title("Parameter 2 in latent logvar space")
    else:
        ax4 = fig.add_subplot(2, 2, 4, projection="3d")
        ax4.set_title("No Parameter 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    plt.show()
    return latent_mus, latent_logvars, param1, param2 if param2 is not None else None

def show_vae_random_reconstructions(
    folder: str,
    label: str,
    model_path: str | None = None,
    model: VAE | None = None,
    latent_dim: int = 6,
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
    latent_dim: int = 6,
    input_dim: int = 2,
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
        ax_in.set_title(f"Input #{idx}, params = [{p_np[0]:+.3f}, {p_np[1]:+.3f}]")
        ax_in.set_xlabel("Feature index")
        ax_in.set_ylabel("log10(I(q))")
        ax_in.grid(True, alpha=0.3)
        ax_in.legend()
        # Reconstruction
        ax_out = axes[idx, 1]
        recon_np, _ = denormalize_generated_Iq(folder, label, recon_np, p_np)
        ax_out.plot(recon_np, "r-", linewidth=1.5, label="Reconstruction")
        p_str = ", ".join([f"{v:+.3f}" for v in p_np])
        ax_out.set_title(f"Recon #{idx}\nParams = [{p_str}]")
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