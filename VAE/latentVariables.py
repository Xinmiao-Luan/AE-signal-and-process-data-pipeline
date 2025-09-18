import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import distance
from scipy.stats import ks_2samp

folder_path = '/Users/xluan3/Desktop/Projects/layer1to5/change latent/result/result'
save_path = folder_path
latent_z = np.load(os.path.join(folder_path, 'latent_z.npy'))
# Compute correlation matrix (8x8)
corr_matrix = np.corrcoef(latent_z.T)
# === LOAD LABELS ===
labels_df = pd.read_csv('/Users/xluan3/Desktop/Projects/layer1to5/classified_with_process_porosity_labels.csv')

## Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Latent Variable Correlation Matrix')
plt.xlabel('Latent Variables')
plt.ylabel('Latent Variables')
plt.tight_layout()
file_name = f'Latent Variable Correlation Matrix.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)


# Run t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
z_2d = tsne.fit_transform(latent_z)
# # save
df_z2d = pd.DataFrame(z_2d, columns=['z2d_1', 'z2d_2'])
df_z2d.to_csv('z_2d_projection.csv', index=False)

## CLASSIFY COLORS
# # Create boolean mask: where Porosity_Label == 1
# is_porosity = labels_df["if it is porosity"] == 1
#
# # Plot
# plt.figure(figsize=(8, 6))
# plt.scatter(z_2d[~is_porosity, 0], z_2d[~is_porosity, 1], c='blue', alpha=0.5, label='Class 3, Porosity=0')
# plt.scatter(z_2d[is_porosity, 0], z_2d[is_porosity, 1], c='red', alpha=0.7, label='Class 3, Porosity=1')
# plt.title("t-SNE of VAE Latent Variables (Class 3)")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(save_path, 'Latent Variable Visualization.png'), dpi=300)

# Plot latent space with select lable
# Create boolean mask: where Porosity_Label == 1
# Create boolean mask where selectLabel == 1
is_selected = labels_df["selectLabel"] == 1

# Extract latent dimensions (assumed as Series or arrays)
z_2d_1 = labels_df["z2d_1"]
z_2d_2 = labels_df["z2d_2"]

# Ensure save_path exists
os.makedirs(save_path, exist_ok=True)

# Plot latent space
plt.figure(figsize=(8, 6))
plt.scatter(z_2d_1[~is_selected], z_2d_2[~is_selected], c='blue', alpha=0.5, label='Cluster = 0', s=6)
plt.scatter(z_2d_1[is_selected], z_2d_2[is_selected], c='red', alpha=0.7, label='Cluster = 1', s=6)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
# plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Latent Variable Visualization with selected clusters.png'), dpi=300)

## only draw original clusters
plt.figure(figsize=(8, 6))
plt.scatter(z_2d[:,0],z_2d[:,1], alpha=0.5)
plt.title("t-SNE of VAE Latent Variables")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Latent Variable Visualization_clusters.png'), dpi=300)

# # plot construction errors
from matplotlib.colors import LogNorm
# labels_df = pd.read_csv('/Users/xluan3/Desktop/Projects/layer1to20/remaining_class3_VAEresults and process data.csv')
# CLASSIFY COLORS
recon_errors = labels_df['reconstruction_error'].values
# Plot
# Clip errors to avoid log(0)
adjusted_errors = np.clip(recon_errors, a_min=1e-3, a_max=None)
# Define how many top points to highlight (e.g., top 1%)
k = int(50)
top_k_indices = np.argsort(adjusted_errors)[-k:]
# Separate data into two sets: highlighted and normal
z_top = z_2d[top_k_indices]
z_rest = np.delete(z_2d, top_k_indices, axis=0)
errors_rest = np.delete(adjusted_errors, top_k_indices)
# Plot
plt.figure(figsize=(9, 6))
# Plot rest of points with YlOrRd + LogNorm
scatter = plt.scatter(
    z_rest[:, 0], z_rest[:, 1],
    c=errors_rest,
    cmap='YlOrRd',
    norm=LogNorm(vmin=np.min(adjusted_errors), vmax=np.max(adjusted_errors)),
    alpha=0.5,
    label='Other points'
)

# Overlay top-k high-error points in solid red
plt.scatter(
    z_top[:, 0], z_top[:, 1],
    color='red',
    edgecolor='black',
    s=30,
    label=f'Top {k} highest errors'
)

plt.colorbar(scatter, label='Reconstruction Error (Log Scale)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Latent Space: Highlighted High Reconstruction Errors')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'Reconstruction Error in Latent Space 50.png'), dpi=300)



## IsolationForest
from sklearn.ensemble import IsolationForest
#model = IsolationForest(contamination=0.2, random_state=42)
model = IsolationForest(contamination=0.05, random_state=42)
labels = model.fit_predict(latent_z)  # -1 = anomaly
# save result
# # save
labels = pd.DataFrame(labels, columns=['isolation result'])
labels.to_csv('isolation result from latent_z_less.csv', index=False)

anomalies = np.where(labels == -1)[0]
print(f"Anomalies detected: {len(anomalies)}")

# Optional visualization

labels_array = labels.values
plt.figure(figsize=(8, 6))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=(labels_array == -1), cmap='coolwarm', alpha=0.5)
plt.title("t-SNE with Anomaly Detection (Isolation Forest)")
file_name = f't-SNE with Anomaly Detection (Isolation Forest)_less.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)

### 2 by 2 dist
df = pd.DataFrame(latent_z, columns=[f'z{i}' for i in range(latent_z.shape[1])])

# Plot pairwise relationships
sns.set(style="ticks")
sns.pairplot(df, plot_kws={'alpha': 0.3, 's': 5})
plt.suptitle("Pairwise Scatter Plots of Latent Variables", y=1.02)
file_name = f'Latent Variable Distribution.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)


## Mahalanobis distance
# Step 1: Compute Mahalanobis distance
mean = np.mean(latent_z, axis=0)
cov = np.cov(latent_z, rowvar=False)
inv_cov = np.linalg.inv(cov)

md = np.array([distance.mahalanobis(z, mean, inv_cov) for z in latent_z])

# Step 2: Anomaly detection based on 99th percentile
threshold = np.percentile(md, 99)
anomalies = np.where(md > threshold)[0]

print(f"{len(anomalies)} anomalies found using Mahalanobis distance.")

# Step 3: t-SNE for visualization
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
z_2d = tsne.fit_transform(latent_z)

# Step 4: Plot t-SNE colored by Mahalanobis-based anomalies
plt.figure(figsize=(10, 8))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c='blue', s=5, alpha=0.5, label='Normal')
plt.scatter(z_2d[anomalies, 0], z_2d[anomalies, 1], c='red', s=10, alpha=0.8, label='Anomaly (Mahalanobis)')
plt.title("t-SNE with Mahalanobis Distance-Based Anomalies")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 5: Save figure
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, "tSNE with with Anomaly Detection (mahalanobis_anomalies).png"), dpi=300)


## correlation check
cols_of_interest = ['theta', 'if it is porosity', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6',  'z7','z8']
correlation_df = labels_df[cols_of_interest].corr()

# Display correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_df,
    annot=True,
    cmap='coolwarm',
    center=0,
    square=True,
    fmt=".2f"
)
plt.title('Correlation between theta, Porosity, and Latent Variables')
plt.tight_layout()
plt.savefig(os.path.join(save_path, "Correlation between theta, Porosity, and Latent Variables.png"), dpi=300)

## more exploration

# split the data with label into two subsets according to the isolation tree' result
base_folder = '/Users/xluan3/Desktop/Projects/layer1to20'
# Corrected file path if needed
input_csv = os.path.join(base_folder, 'remaining_class3_VAEresults and process data.csv')
# Load and split
df = pd.read_csv(input_csv)
df_1 = df[df['isolation result_less'] == 1]
df_abnormal = df[df['isolation result_less'] == -1]

df_1.to_csv(os.path.join(base_folder, 'class3_VAEresults_isolation_1_less.csv'), index=False)
df_abnormal.to_csv(os.path.join(base_folder, 'class3_VAEresults_isolation_abnormal_less.csv'), index=False)

# plot and see the dist between the
# Columns to plot
columns_to_plot = [
    'if it is porosity', 'Power [W]', 'rho', 'phi', 'theta',
]
available_columns = [col for col in columns_to_plot if col in df.columns]
output_dir = '/Users/xluan3/Desktop/Projects/layer1to20/result/result'
# Function to plot pairplot
def plot_and_save(dataframe, title, filename):
    if not available_columns:
        print(f"No valid columns found for {title}")
        return
    g = sns.pairplot(dataframe[available_columns], diag_kind='kde', plot_kws={'alpha': 0.5}, height = 3, aspect = 1.2)
    g.fig.set_size_inches(20, 20)  # Overall figure size
    g.fig.suptitle(title, y=1.02)
    g.fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {filename}")
# Plotting
output_dir = '/Users/xluan3/Desktop/Projects/layer1to20/result/result'
# plot_and_save(df, "Pairwise Relationships - Full Dataset", "pairplot_full_dataset.png")
plot_and_save(df_1, "Pairwise Relationships - Isolation Result == 1", "less_pairplot_isolation_1.png")
plot_and_save(df_abnormal, "Pairwise Relationships - Isolation Result == -1", "lesspairplot_isolation_abnormal.png")

# KS test
# KS test for rho
ks_rho = ks_2samp(df_1['rho'], df_abnormal['rho'])
print(f"KS Test for rho → statistic: {ks_rho.statistic:.4f}, p-value: {ks_rho.pvalue:.4f}")
# KS test for phi
ks_phi = ks_2samp(df_1['phi'], df_abnormal['phi'])
print(f"KS Test for phi → statistic: {ks_phi.statistic:.4f}, p-value: {ks_phi.pvalue:.4f}")

# Plot cumulative distributions and save figure
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df_1['rho'].sort_values().reset_index(drop=True).plot(label='df_1')
df_abnormal['rho'].sort_values().reset_index(drop=True).plot(label='df_abnormal')
plt.title('Rho Cumulative Distribution')
plt.legend()

plt.subplot(1, 2, 2)
df_1['phi'].sort_values().reset_index(drop=True).plot(label='df_1')
df_abnormal['phi'].sort_values().reset_index(drop=True).plot(label='df_abnormal')
plt.title('Phi Cumulative Distribution')
plt.legend()

plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_dir, 'less_KS_test_rho_phi_cumulative_distribution.png')
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"Plot saved to: {plot_path}")

