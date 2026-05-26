import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")

file_path = r'your_path_here\clean_element_stable.csv'
df = pd.read_csv(file_path)

features_list = ['volume_per_atom', 'mean MeltingT', 'mean CovalentRadius', 
                 'range Electronegativity', 'mean Electronegativity', 
                 'mean NpValence', 'mean NdValence', 'range Polarizability', 'mean Polarizability']
target = 'formation_energy_per_atom'

X = df[features_list]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LGBMRegressor(random_state=42, verbose=-1)
model.fit(X_scaled, y)

element_props = {
    'Li': [453.6, 1.28, 0.98, 0, 0, 24.3],
    'Mg': [923.0, 1.41, 1.31, 0, 0, 10.6],
    'Al': [933.47, 1.21, 1.61, 1, 0, 6.8],
    'Ni': [1728.0, 1.24, 1.91, 0, 8, 6.8],
    'Cu': [1357.7, 1.32, 1.90, 0, 10, 6.7],
    'Zn': [692.68, 1.22, 1.65, 0, 10, 5.7],
    'Ag': [1234.9, 1.45, 1.93, 0, 10, 7.2],
    'In': [429.75, 1.42, 1.78, 1, 10, 10.1],
    'Sn': [505.08, 1.40, 1.96, 2, 10, 12.0],
    'Bi': [544.7, 1.48, 2.02, 3, 10, 11.5]
}

target_metals = [m for m in element_props.keys() if m != 'Li']
avg_vol = df['volume_per_atom'].mean()

step_val = 0.097
fractions = np.arange(0.05, 1.00, step_val)
step_097_results = []

for m in target_metals:
    m_energies = []
    li_p, m_p = element_props['Li'], element_props[m]
    
    for x in fractions:
        mean_f = x * np.array(li_p) + (1-x) * np.array(m_p)
        row = [avg_vol, mean_f[0], mean_f[1], abs(li_p[2]-m_p[2]), 
               mean_f[2], mean_f[3], mean_f[4], abs(li_p[5]-m_p[5]), mean_f[5]]
        
        X_new = pd.DataFrame([row], columns=features_list)
        m_energies.append(model.predict(scaler.transform(X_new))[0])
    
    step_097_results.append({
        'Metal': m,
        'Average_FE': np.mean(m_energies),
        'Min_FE': np.min(m_energies)
    })

res_097 = pd.DataFrame(step_097_results).sort_values(by='Average_FE')

print(f"\n--- Rank ---")
print(res_097[['Metal', 'Average_FE', 'Min_FE']].to_string(index=False))

plt.figure(figsize=(10, 6))
plt.rcParams['font.sans-serif'] = ['Arial']
sns.barplot(data=res_097, x='Metal', y='Average_FE', palette='viridis')
plt.title(f'Metal Stability Ranking ', fontsize=14)
plt.ylabel('Average Formation Energy (eV/atom)')
plt.show()
