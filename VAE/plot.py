import os
import numpy as np
import matplotlib.pyplot as plt

# Load data
folder_path = '/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/Spectro/Class_1/result'
raw_data = np.load(os.path.join(folder_path, 'ae_data.npy'))  # Shape: (23193, 200)
reconstructed = np.load(os.path.join(folder_path, 'reconstructed.npy'))  # Shape: (23193, 200)

# draw heatmaps
# Parameters
window_size = 300
stride = 100
num_to_plot = 5
cmap_choice = 'jet'  # Use 'turbo' if you want better perceptual contrast
save_path = folder_path


for i in range(num_to_plot):
    start = i * stride
    end = start + window_size
    if end > raw_data.shape[0]:
        break

    raw_window = raw_data[start:end, :].T
    recon_window = reconstructed[start:end, :].T

    # Use percentile-based contrast
    vmin = np.percentile(raw_window, 5)
    vmax = np.percentile(raw_window, 99)

    # Create figure with extra space for colorbar
    fig, axs = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'right': 0.9}, sharex=True)

    im0 = axs[0].imshow(raw_window, aspect='auto', origin='lower',
                        cmap=cmap_choice, vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Raw Spectrogram #{i}')
    axs[0].set_ylabel('Frequency Bin')

    im1 = axs[1].imshow(recon_window, aspect='auto', origin='lower',
                        cmap=cmap_choice, vmin=vmin, vmax=vmax)
    axs[1].set_title(f'Reconstructed Spectrogram #{i}')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Frequency Bin')

    # Create outer colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])  # Adjusted to be fully outside
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label('Amplitude')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space on right for colorbar

    # Save figure
    file_name = f'spectrogram_comparison_final_{i}.png'
    plt.savefig(os.path.join(save_path, file_name), dpi=300)
    plt.close()

# # latent_z
# Example latent_z array, shape: (23913, 8)
# Replace this with your actual latent_z data
latent_z = np.load(os.path.join(folder_path, 'latent_z.npy'))

# Set up the figure
plt.figure(figsize=(14, 8))

# Plot each latent variable in a different color
for i in range(latent_z.shape[1]):  # There are 8 latent variables
    plt.plot(latent_z[:, i], label=f'Latent Variable {i+1}')

# Add labels, title, and legend
plt.title('Latent Variables over Time')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.tight_layout()
file_name = f'latent variables.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)

# draw only the first 99
# Set up the figure
latent_z_fist99 = latent_z[0:100,:]
plt.figure(figsize=(14, 8))

# Plot each latent variable in a different color
for i in range(latent_z_fist99.shape[1]):  # There are 8 latent variables
    plt.plot(latent_z_fist99[:, i], label=f'Latent Variable {i+1}')

# Add labels, title, and legend
plt.title('Latent Variables over Time')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

# Show plot
plt.tight_layout()
file_name = f'latent variables_first99.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)

