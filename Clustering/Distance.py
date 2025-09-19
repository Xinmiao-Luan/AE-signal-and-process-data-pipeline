from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

# Fit multivariate Gaussian to latent space
mean = np.mean(latent_z, axis=0)
cov = np.cov(latent_z, rowvar=False)
inv_cov = np.linalg.inv(cov)

# Compute Mahalanobis distance for each point
md = np.array([distance.mahalanobis(z, mean, inv_cov) for z in latent_z])

# Threshold: 95th or 99th percentile
threshold = np.percentile(md, 99)
anomalies = np.where(md > threshold)[0]

print(f"{len(anomalies)} anomalies found")

# plot
save_path = '/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/Spectro/Class_1/result'
plt.figure(figsize=(12, 4))
plt.plot(md, label='Mahalanobis Distance')
plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
plt.scatter(anomalies, md[anomalies], color='red', label='Anomalies')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Distance')
plt.title('Latent-based Anomaly Detection')
plt.tight_layout()
file_name = f'anomaly detection_distance.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)