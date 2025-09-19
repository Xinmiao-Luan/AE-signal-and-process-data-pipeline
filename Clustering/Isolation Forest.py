import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest



# Fit Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
labels = model.fit_predict(latent_z)  # -1 = anomaly, 1 = normal

# Get anomaly indices
anomalies = np.where(labels == -1)[0]
print(f"{len(anomalies)} anomalies found")

# Plot result
plt.figure(figsize=(14, 4))
plt.plot(range(len(labels)), labels, label='IF labels', alpha=0.2)
plt.scatter(anomalies, [-1]*len(anomalies), color='red', label='Anomalies', s=10)

plt.axhline(y=-1, color='gray', linestyle='--', linewidth=1)
plt.title('Anomaly Detection using Isolation Forest on Latent Space')
plt.xlabel('Time Step')
plt.ylabel('Label')
plt.yticks([-1, 1], ['Anomaly', 'Normal'])
plt.legend()
plt.tight_layout()
file_name = f'anomaly detection_isolation forest.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)
