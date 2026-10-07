import matplotlib.pyplot as plt
import pandas as pd
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
    "figure.dpi": 300,
})

colors_cond = ['#9ecae1', '#fd8d3c', '#d62728'] # Lesion_Thal, Lesion_Kappa, Intact

df_D = pd.read_csv('data/master_D.csv')
df_E = pd.read_csv('data/master_E.csv')

cond_order = ["Lesion_Thal", "Lesion_Kappa", "Intact"]
var_order = ["Value", "State", "Volatility", "Action"]

df_D['Condition'] = pd.Categorical(df_D['Condition'], categories=cond_order, ordered=True)
df_E['Condition'] = pd.Categorical(df_E['Condition'], categories=cond_order, ordered=True)

for var in var_order:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=False)
    
    # Cortex
    ax = axes[0]
    sub_D = df_D[df_D['Variable'] == var]
    sns.boxplot(data=sub_D, x='Condition', y='Efficiency', palette=colors_cond, ax=ax, showfliers=False, width=0.5)
    sns.stripplot(data=sub_D, x='Condition', y='Efficiency', color='black', alpha=0.3, jitter=True, size=4, ax=ax)
    ax.set_title(f"Cortical Efficiency ({var})")
    ax.set_xlabel('')
    ax.set_xticklabels(['Thal. Lesion', 'Kappa Lesion', 'Intact'], rotation=30)
    ax.set_ylabel('Efficiency')
    
    # Cerebellum
    ax = axes[1]
    sub_E = df_E[df_E['Variable'] == var]
    sns.boxplot(data=sub_E, x='Condition', y='Efficiency', palette=colors_cond, ax=ax, showfliers=False, width=0.5)
    sns.stripplot(data=sub_E, x='Condition', y='Efficiency', color='black', alpha=0.3, jitter=True, size=4, ax=ax)
    ax.set_title(f"Cerebellar Efficiency ({var})")
    ax.set_xlabel('')
    ax.set_xticklabels(['Thal. Lesion', 'Kappa Lesion', 'Intact'], rotation=30)
    ax.set_ylabel('Efficiency')
    
    plt.tight_layout()
    fig.savefig(f'presentations/plots/plot_ablation_{var.lower()}.png')
    plt.close(fig)

print("Generated 4 split plots successfully.")
