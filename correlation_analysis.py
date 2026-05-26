import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import os

# 1. Set your file path
file_path = r'your_path_here\data.csv'
df = pd.read_csv(file_path)

df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]

target = 'formation_energy_per_atom'
label = 'formula_pretty'

X_raw = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
y = df[target]

print(f"number of original features: {X_raw.shape[1]}")

# ---Step 1: Mutual Information (MI) Filtering ---
print("Calculating Mutual Information (MI) scores...")
mi_scores = mutual_info_regression(X_raw, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X_raw.columns).sort_values(ascending=False)
informative_features = mi_series[mi_series > 0.001].index.tolist()
X_informative = X_raw[informative_features]
print(f"Number of features retained after MI filtering: {len(informative_features)}")

# --- Step 2: Pearson Correlation Coefficient (r) Analysis and Heatmap ---
corr_matrix = X_informative.corr()
plt.figure(figsize=(18, 14))
sns.heatmap(
    corr_matrix, 
    annot=True,          
    cmap='RdBu_r',       
    fmt=".2f",           
    center=0, 
    linewidths=.5, 
    square=True,
    cbar_kws={"shrink": .8}
)
plt.title('Correlation Heatmap of Informative Features (|r| Analysis)', fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()  

upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop = [c for c in to_drop if 'Polarizability' not in c]

X_final = X_informative.drop(columns=to_drop)

print(f"\nCorrelation redundancy elimination (|r| > 0.95): {to_drop}")
print(f"Final number of features used for modeling: {X_final.shape[1]}")
print(f"Final feature list: {X_final.columns.tolist()}")

# Save result
# final_df = pd.concat([df[[label, 'symmetry']], X_final, y], axis=1)
# final_df.to_excel('Selected_Features_Final.xlsx', index=False)
