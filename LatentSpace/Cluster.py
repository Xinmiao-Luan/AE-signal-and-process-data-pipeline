import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import hdbscan
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# === Step 1: Load your 8D latent variables ===
latent_z = pd.read_csv('/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/Spectro/Class_3/result/latent_z.csv')  # replace with your actual file path
latent_z.columns = [f'z{i+1}' for i in range(latent_z.shape[1])]  # optional: name columns z1–z8
save_path = '/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/Spectro/Class_3/result'

# # === Step 2: Normalize (important for DBSCAN/HDBSCAN) ===
# scaler = StandardScaler()
# latent_scaled = scaler.fit_transform(latent_z)

# # # DBSCAN
# dbscan = DBSCAN(eps=2, min_samples=42)
# db_labels = dbscan.fit_predict(latent_z)
# min_cluster = 20 # Try different values: 5, 10, 20, 50, 100
# hdb = hdbscan.HDBSCAN(min_cluster_size = min_cluster)
# hdb_labels = hdb.fit_predict(latent_z)
# # # plot
# plt.figure(figsize=(8, 6))
# plt.scatter(z_2d[:, 0], z_2d[:, 1], c=hdb_labels, cmap='coolwarm', s=5)
# plt.title("HDBSCAN Clustering (n=2)")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.tight_layout()
# plt.savefig("HDBSCAN.png", dpi=300)

# K means
# meanskmeans = KMeans(n_clusters=2, random_state=42)
# # kmeans_labels = kmeans.fit_predict(latent_z)
# kmeans_labels = kmeans.fit_predict(z_2d)


#  Agglomerative Clustering
# too slow
# agglo = AgglomerativeClustering(n_clusters=2)
# agg_labels = agglo.fit_predict(latent_z)
# np.bincount(agg_labels)
n_clusters = 2  # or more if you expect more groups
agg_single = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
labels_single = agg_single.fit_predict(z_2d)


# plot
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(latent_z)
# # save
df_z2d = pd.DataFrame(z_2d, columns=['z2d_1', 'z2d_2'])
df_z2d.to_csv('z_2d_projection.csv', index=False)

plt.figure(figsize=(8, 6))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c = labels_single, cmap='coolwarm', s=5)
plt.title("Agglomerative_latent_clusters (n=2)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig("Agglomerative_latent_clusters.png", dpi=300)