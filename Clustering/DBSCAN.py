import matplotlib.pyplot as plt
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
file_name = f'anomaly detection.png'
plt.savefig(os.path.join(save_path, file_name), dpi=300)