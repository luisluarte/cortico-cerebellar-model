import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

os.makedirs('presentations/plots', exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "font.size": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.5,
    "figure.dpi": 300,
    "figure.figsize": (8, 6)
})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 1. NSGA-II Hypervolume Decay
try:
    df_hv = pd.read_csv('data/gamma_hv_bootstrap_4d_strict.csv')
    # Filter for Mcc (bio) and heuristics
    models = df_hv['Model'].unique()
    fig, ax = plt.subplots()
    
    for m, c, l in zip(['bio', 'q', 'wsls'], [colors[0], colors[1], colors[2]], ['$M_{cc}$', 'Q-Learning', 'WSLS']):
        sub = df_hv[df_hv['Model'] == m]
        ax.plot(sub['Gamma'], sub['HV_mean'], label=l, color=c, linewidth=3)
        ax.fill_between(sub['Gamma'], sub['HV_lwr'], sub['HV_upr'], color=c, alpha=0.2)
    
    ax.set_xlabel(r'Autonomy ($\gamma$)')
    ax.set_ylabel(r'4D Hypervolume ($\mu$)')
    ax.set_title('Performance Decay across Autonomy Spectrum')
    ax.legend()
    plt.tight_layout()
    fig.savefig('presentations/plots/plot_nsga2_hv.png')
    plt.close(fig)
except Exception as e:
    print(f"Failed HV decay: {e}")

# 2. Memory Capacity Ablation (Raw points + Boxplot)
try:
    df_mc = pd.read_csv('data/mc_ablation_df.csv')
    df_sub = df_mc[df_mc['Ablation_Level'].isin([0, 90, 100])]
    fig, ax = plt.subplots()
    sns.boxplot(data=df_sub, x='Ablation_Level', y='Total_MC', color='lightgray', ax=ax, showfliers=False)
    sns.stripplot(data=df_sub, x='Ablation_Level', y='Total_MC', color=colors[0], alpha=0.6, jitter=True, ax=ax)
    ax.set_xlabel('Thalamic Ablation (%)')
    ax.set_ylabel('Total Memory Capacity (MC)')
    ax.set_title('Fading Memory Collapse')
    plt.tight_layout()
    fig.savefig('presentations/plots/plot_memory_capacity.png')
    plt.close(fig)
except Exception as e:
    print(f"Failed MC Ablation: {e}")

# 3. PSIS-LOO
fig, ax = plt.subplots()
models = [r'Unconstrained' + '\n' + r'(Flat Prior)', r'Guided' + '\n' + r'($\sigma=e^{-0.5}$)', r'Zero-Shot' + '\n' + r'($\sigma=e^{-1.5}$)', 'WSLS', 'Q-Learning']
elpd = [1712.0, 1548.6, 1511.4, 1212.7, 1200.4]
se = [4.5, 4.9, 5.0, 5.2, 3.7]
x = np.arange(len(models))
ax.errorbar(x, elpd, yerr=se, fmt='o', color=colors[0], capsize=5, capthick=2, markersize=10)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha='right')
ax.set_ylabel('PSIS-LOO ELPD')
ax.set_title('Out-of-Sample Generalization')
plt.tight_layout()
fig.savefig('presentations/plots/plot_psis_loo.png')
plt.close(fig)

# 4. State Readout Efficiency (Raw points + Boxplot)
try:
    df_eff = pd.read_csv('data/readout_efficiency_df.csv')
    fig, ax = plt.subplots()
    sns.boxplot(data=df_eff, x='Ablation_Pct', y='Efficiency', color='lightgray', ax=ax, showfliers=False)
    sns.stripplot(data=df_eff, x='Ablation_Pct', y='Efficiency', color=colors[3], alpha=0.6, jitter=True, ax=ax)
    ax.set_xlabel(r'Thalamic Ablation (%)')
    ax.set_ylabel(r'Cortical Readout Efficiency ($R^2 / V$)')
    ax.set_title('Synthetic Latent Deficits')
    plt.tight_layout()
    fig.savefig('presentations/plots/plot_efficiency.png')
    plt.close(fig)
except Exception as e:
    print(f"Failed Efficiency: {e}")

# 5. Differential Entropy (Raw points + Boxplot)
try:
    df_ent = pd.read_csv('data/diff_entropy_ready.csv')
    df_sub_ent = df_ent[df_ent['Ablation_Pct'].isin([0, 90, 100])]
    fig, ax = plt.subplots()
    sns.boxplot(data=df_sub_ent, x='Ablation_Pct', y='Total_Diff_Entropy', color='lightgray', ax=ax, showfliers=False)
    sns.stripplot(data=df_sub_ent, x='Ablation_Pct', y='Total_Diff_Entropy', color=colors[4], alpha=0.6, jitter=True, ax=ax)
    ax.set_xlabel('Thalamic Ablation (%)')
    ax.set_ylabel('Differential Entropy (nats)')
    ax.set_title('Policy Perseverance & Stochastic Volatility')
    plt.tight_layout()
    fig.savefig('presentations/plots/plot_entropy.png')
    plt.close(fig)
except Exception as e:
    print(f"Failed Entropy: {e}")

