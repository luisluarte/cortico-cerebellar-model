import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/behavioral_compilate.csv')
df['rt'] = (df['ttr'] - df['ttp']) / 1000
df['iti'] = (df['ttp'] - df.groupby('participant_id')['ttF'].shift(1)) / 1000

subs = df['participant_id'].unique()[:30]
df_sub = df[df['participant_id'].isin(subs)].copy()

df_sub['iti'] = df_sub['iti'].fillna(df_sub['iti'].median())
df_sub.loc[df_sub['iti'] < 0, 'iti'] = df_sub['iti'].median()

df_sub = df_sub[(df_sub['Resp'].isin([1, 2])) & (df_sub['rt'] > 0.1)].copy()

subj_map = {s: i+1 for i, s in enumerate(df_sub['participant_id'].unique())}

stan_data = {
    'N': len(df_sub),
    'N_subj': len(subj_map),
    'subj': df_sub['participant_id'].map(subj_map).tolist(),
    'Bd1': (df_sub['Bd1'] - 1).tolist(),
    'Bd2': (df_sub['Bd2'] - 1).tolist(),
    'Resp': df_sub['Resp'].tolist(),
    'Reward': df_sub['F'].tolist(),
    'RT': df_sub['rt'].tolist(),
    'ITI': df_sub['iti'].tolist()
}

with open('data/stan_data_N30.json', 'w') as f:
    json.dump(stan_data, f)

print('Data prep done.')
