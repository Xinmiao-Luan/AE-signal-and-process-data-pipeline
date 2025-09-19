# Ref:
# https://medium.com/@sofeikov/implementing-variational-autoencoders-from-scratch-533782d8eb95

import os
import scipy.io
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Load and preprocess AE data
# data dim: frequency x time stamp x # channel
# 200 x 99 x 4 or 200 x 98 x 4
# for all channels
def load_mat_ae_data(folder_path):
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            mat = scipy.io.loadmat(os.path.join(folder_path, file))
            for key in mat:
                if isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
                    arr = mat[key]
                    if arr.shape[1] in [98, 99] and arr.shape[2] == 4:
                        arr = arr[:, :98, :]            # Crop to 98
                        arr = np.abs(arr)               # Take magnitude
                        arr = arr.reshape(arr.shape[0], -1)  # Flatten to (200, 392)
                        data_list.append(arr)
    if not data_list:
        raise ValueError("No valid .mat files found in the folder.")
    return np.concatenate(data_list, axis=0).astype(np.float32)
    
# for one channel
def load_mat_ae_data_freq_bin_view(folder_path, channel_idx=0):
    data_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".mat"):
            mat = scipy.io.loadmat(os.path.join(folder_path, file))
            for key in mat:
                if isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
                    arr = mat[key]  # shape (200, 98/99, 4)
                    if arr.shape[1] in [98, 99] and arr.shape[2] == 4:
                        arr = arr[:, :98, channel_idx]  # shape (200, 98)
                        arr = np.abs(arr)
                        arr = arr.T  # shape (98, 200)
                        data_list.append(arr)
    if not data_list:
        raise ValueError("No valid .mat files found in the folder.")
    return np.concatenate(data_list, axis=0).astype(np.float32)  # shape: (98*num_files, 200)


# 2. VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=200, latent_dim=2):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# 3. Define loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

# 4. Train the VAE
def train_vae_model(data, result_dir, input_dim=200, latent_dim=2, lr=1e-3, epochs=20, batch_size=32):
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.from_numpy(data)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VAE(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")

    torch.save(model.state_dict(), os.path.join(result_dir, "vae_model.pth"))
    return model

# 5. Save Results
# vae_model.pth - trained VAE model
# latent_z.npy - latent, (num_samples, latent_dim)
# reconstructed.npy - from decoder, (N, 392)
# reconstruction.png

def save_results(model, ae_data, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        ae_tensor = torch.from_numpy(ae_data)
        recon, mu, logvar = model(ae_tensor)
        z = model.reparameterize(mu, logvar)
        # Compute reconstruction error (per sample MSE)
        reconstruction_errors = F.mse_loss(recon, ae_tensor, reduction='none')  # shape: (N, 200)
        reconstruction_errors = reconstruction_errors.mean(dim=1).numpy()  # shape: (N,)

    np.save(os.path.join(result_dir, "latent_z.npy"), z.numpy())
    np.save(os.path.join(result_dir, "ae_data.npy"), ae_data)
    np.save(os.path.join(result_dir, "reconstructed.npy"), recon.numpy())
    np.save(os.path.join(result_dir, "reconstruction_errors.npy"), reconstruction_errors)
    # .csv
    np.savetxt(os.path.join(result_dir, "latent_z.csv"), z.numpy(), delimiter=",")
    np.savetxt(os.path.join(result_dir, "ae_data.csv"), ae_data, delimiter=",")
    np.savetxt(os.path.join(result_dir, "reconstructed.csv"), recon.numpy(), delimiter=",")
    np.savetxt(os.path.join(result_dir, "reconstruction_errors.csv"), reconstruction_errors, delimiter=",")

## 6. Run
folder_path = '/Users/xluan3/Desktop/Projects/layer1to5/spectrograms'
save_path = '/Users/xluan3/Desktop/Projects/layer1to5/change latent/result'
ae_data = load_mat_ae_data_freq_bin_view(folder_path,0)
ae_data_select = ae_data
result_dir = os.path.join(save_path, "result")
vae_model = train_vae_model(ae_data_select, result_dir)
save_results(vae_model, ae_data_select, result_dir)

# save ae
np.savetxt(os.path.join(save_path,'ae_raw_data_VAE_input.csv'), ae_data, delimiter=",")

# # check the size of latent_z and reconstructed
latent_z = np.load(os.path.join(save_path, 'result', 'latent_z.npy'))
print(latent_z.shape)
reconstructed = np.load(os.path.join(save_path, 'result', 'reconstructed.npy'))
print(reconstructed.shape)
