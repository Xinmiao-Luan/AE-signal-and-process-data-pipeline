import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Load data
df = pd.read_csv('/Users/xluan3/Desktop/Projects/layer1to20/remaining_class3_VAEresults and process data.csv')

# Identify top 50 samples with highest reconstruction errors
top_50_high_error_samples = df.nlargest(50, 'reconstruction_errors')

# Mark high error flag
df['high_error_flag'] = df['Index'].isin(top_50_high_error_samples['Index']).astype(int)

# Select process parameters for analysis
process_parameters = ['Power [W]', 'theta', 'rho', 'phi']

# Plot distributions using histograms
for param in process_parameters:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=param, hue='high_error_flag', element="step", stat="density", common_norm=False, alpha=0.5)
    plt.title(f"Histogram of {param} (Top 50 High-Error Samples vs. Others)")
    plt.legend(title='High Error Flag', labels=['Other Samples', 'Top 50 High Error'])
    plt.tight_layout()
    plt.show()

# Statistical tests using Mann-Whitney U test
# results = []
# for param in process_parameters:
#     group_high_error = df[df['high_error_flag'] == 1][param].dropna()
#     group_others = df[df['high_error_flag'] == 0][param].dropna()
#
#     stat, p_value = mannwhitneyu(group_high_error, group_others, alternative='two-sided')
#
#     results.append({
#         'Parameter': param,
#         'High Error Median': group_high_error.median(),
#         'Others Median': group_others.median(),
#         'Mann-Whitney U Statistic': stat,
#         'p-value': p_value
#     })

results_df = pd.DataFrame(results)
print(results_df)

# Optional: save results
results_df.to_csv('statistical_test_results.csv', index=False)

# Identify the top 50 samples with the highest reconstruction errors
top_50_high_error_samples = df.nlargest(50, 'reconstruction_errors')
other_samples = df[~df['Index'].isin(top_50_high_error_samples['Index'])]
# Export the result to CSV
top_50_high_error_samples.to_csv(os.path.join(save_path,'top_50_high_reconstruction_error_samples.csv'), index=False)
other_samples.to_csv(os.path.join(save_path,'other_samples.csv'), index=False)