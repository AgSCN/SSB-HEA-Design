import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from itertools import combinations
import warnings

warnings.filterwarnings("ignore")

# 1. Model Training Process
file_path = r'your_path_here\clean_element_stable.csv'
df = pd.read_csv(file_path)
features_list = ['volume_per_atom', 'mean MeltingT', 'mean CovalentRadius', 
                 'range Electronegativity', 'mean Electronegativity', 
                 'mean NpValence', 'mean NdValence', 'range Polarizability', 'mean Polarizability']
target = 'formation_energy_per_atom'

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_list])
model = LGBMRegressor(random_state=42, verbose=-1).fit(X_scaled, df[target])

# 2. Definition of Elemental Properties
element_props = {
    'Li': [453.6, 1.28, 0.98, 0, 0, 24.3],
    'Mg': [923.0, 1.41, 1.31, 0, 0, 10.6], 'Al': [933.47, 1.21, 1.61, 1, 0, 6.8],
    'Ni': [1728.0, 1.24, 1.91, 0, 8, 6.8], 'Cu': [1357.7, 1.32, 1.90, 0, 10, 6.7],
    'Zn': [692.68, 1.22, 1.65, 0, 10, 5.7], 'Ag': [1234.9, 1.45, 1.93, 0, 10, 7.2],
    'In': [429.75, 1.42, 1.78, 1, 10, 10.1], 'Sn': [505.08, 1.40, 1.96, 2, 10, 12.0],
    'Bi': [544.7, 1.48, 2.02, 3, 10, 11.5]
}

candidates = ['Bi', 'Sn', 'In', 'Ag', 'Zn', 'Cu', 'Ni', 'Al', 'Mg']
avg_vol = df['volume_per_atom'].mean()

step_val = 0.43
fractions = np.arange(0.05, 1.00, step_val)

# 3. Iterative Calculation
hea_combinations = list(combinations(candidates, 5))
hea_results = []


for combo in hea_combinations:
    combo_props = np.mean([element_props[m] for m in combo], axis=0)
    m_energies = []
    li_p = element_props['Li']
    
    for x in fractions:
        mean_f = x * np.array(li_p) + (1-x) * combo_props
        all_elements = list(combo) + ['Li']
        elecs = [element_props[e][2] for e in all_elements]
        polars = [element_props[e][5] for e in all_elements]
        range_e, range_p = max(elecs) - min(elecs), max(polars) - min(polars)
        
        row = [avg_vol, mean_f[0], mean_f[1], range_e, mean_f[2], 
               mean_f[3], mean_f[4], range_p, mean_f[5]]
        
        X_new_scaled = scaler.transform(pd.DataFrame([row], columns=features_list))
        m_energies.append(model.predict(X_new_scaled)[0])
    
    hea_results.append({
        'Combination': '-'.join(combo),
        'Avg_Formation_Energy': np.mean(m_energies),
        'Min_Formation_Energy': np.min(m_energies),
        'Stable_Li_Ratio': round(fractions[np.argmin(m_energies)], 3)
    })

# 4. Results Presentation 
hea_df = pd.DataFrame(hea_results).sort_values(by='Avg_Formation_Energy')

print("\n--- Top 10 Average Single-Atom Formation Energies ---")
print(hea_df.head(10).to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

# Check Target Combination Ranking
target_combo = "Bi-Sn-In-Ag-Zn"
if target_combo in hea_df['Combination'].head(1).values:
    print(f"\nSucceeded！Target combination {target_combo} ranks first.")
else:
    rank = hea_df[hea_df['Combination'] == target_combo].index[0] + 1
    print(f"\nNote: {target_combo} is currently ranked {rank}.")
