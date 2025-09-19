from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Assume latent_z is your latent variable matrix from VAE, shape (n_samples, latent_dim)
# Example: latent_z = vae_model.encoder(x)

# Step 1: Apply PCA
pca = PCA()
pca.fit(latent_z)
# Step 2: Extract explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = explained_variance.cumsum()

# Step 3: Plot
save_path = '/Users/xluan3/Desktop/Projects/layer1to20/result/result'
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA on VAE Latent Variables')
plt.grid(True)
plt.savefig(os.path.join(save_path,"PCA on VAE Latent Variables.png"), dpi=300)

# Optional: Save explained variance to CSV
explained_variance_df = pd.DataFrame({
    'PC': range(1, len(explained_variance) + 1),
    'Explained Variance Ratio': explained_variance,
    'Cumulative Variance': cumulative_variance
})
explained_variance_df.to_csv(os.path.join(save_path,'latent_variable_pca_explained_variance.csv'), index=False)