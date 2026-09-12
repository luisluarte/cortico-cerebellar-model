import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec

# Publication styling
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300
})

# Load the data
df = pd.read_csv('results/pruning_final_pop.csv')

# Find the absolute best model
best_idx = df['Loss'].idxmin()
best_granule = df.loc[best_idx, 'Granule']
best_dcn = df.loc[best_idx, 'DCN']
best_loss = df.loc[best_idx, 'Loss']

# Create figure and GridSpec
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 3, wspace=0.1, hspace=0.1)

# Main scatter plot (Granule vs DCN colored by Loss)
ax_main = fig.add_subplot(gs[1:3, 0:2])
scatter = ax_main.scatter(df['Granule'], df['DCN'], 
                         c=df['Loss'], cmap='viridis_r', 
                         s=80, alpha=0.8, edgecolor='white', linewidth=0.5)

# Highlight the optimum
ax_main.scatter(best_granule, best_dcn, color='red', marker='*', s=300, 
                edgecolor='black', linewidth=1, label='Global Optimum\n(GC: 836, DCN: 362)', zorder=5)

ax_main.set_xlabel('Granule Layer Size ($N_G$)')
ax_main.set_ylabel('DCN Layer Size ($N_{DCN}$)')
ax_main.set_xlim(600, 1000)
ax_main.set_ylim(200, 400)
ax_main.grid(True, linestyle='--', alpha=0.5)
ax_main.legend(loc='lower left')

# Top Marginal (Granule Distribution)
ax_top = fig.add_subplot(gs[0, 0:2], sharex=ax_main)
sns.kdeplot(data=df, x='Granule', ax=ax_top, fill=True, color='teal', alpha=0.6)
ax_top.axvline(best_granule, color='red', linestyle='--', linewidth=2)
ax_top.set_ylabel('Density')
ax_top.tick_params(axis="x", labelbottom=False)
ax_top.set_xlabel('')
ax_top.spines['top'].set_visible(False)
ax_top.spines['right'].set_visible(False)

# Right Marginal (DCN Distribution)
ax_right = fig.add_subplot(gs[1:3, 2], sharey=ax_main)
sns.kdeplot(data=df, y='DCN', ax=ax_right, fill=True, color='teal', alpha=0.6)
ax_right.axhline(best_dcn, color='red', linestyle='--', linewidth=2)
ax_right.set_xlabel('Density')
ax_right.tick_params(axis="y", labelleft=False)
ax_right.set_ylabel('')
ax_right.spines['top'].set_visible(False)
ax_right.spines['right'].set_visible(False)

# Add Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.5])
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Distillation Loss (RMSE)', rotation=270, labelpad=20)

plt.suptitle('Evolutionary Discovery of Cerebellar Topology', y=0.95, fontweight='bold')

plt.savefig('results/topology_discovery.png', bbox_inches='tight', dpi=300)
plt.savefig('results/topology_discovery.pdf', bbox_inches='tight')
print("Saved publication figures to results/topology_discovery.png and .pdf")
