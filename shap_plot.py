import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

# 1. Load Data
file_path = r'your_path_here\clean_data1.csv'
df = pd.read_csv(file_path)

# 2. Define Features and Target
target = 'formation_energy_per_atom'
# Exclude target and non-numerical 'formula_pretty'
features = [col for col in df.columns if col not in [target, 'formula_pretty']]

X = df[features]
y = df[target]

# 3. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 4. Train Model
# Using raw features (without scaling) makes SHAP values easier to interpret physically
model = LGBMRegressor(random_state=42, verbose=-1)
model.fit(X_train, y_train)

# 5. SHAP Explanation Analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# --- Plotting Configuration ---

# Set font to Arial and ensure English labels
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False 

# Figure 1: Summary Plot (Distribution of feature influence)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Summary Plot: Feature Influence on Formation Energy", fontsize=14, fontname='Arial')
plt.xlabel("SHAP Value (impact on model output)", fontname='Arial')
plt.show()

# Figure 2: Bar Plot (Global Importance Ranking)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Global Importance: Average Absolute Contribution", fontsize=14, fontname='Arial')
plt.xlabel("Mean |SHAP Value| (average impact magnitude)", fontname='Arial')
plt.show()

# Figure 3: Dependence Plot (Top Feature)
top_feature = X.columns[np.argsort(np.abs(shap_values).mean(0))[-1]]
plt.figure(figsize=(10, 6))
shap.dependence_plot(top_feature, shap_values, X_test, show=False)
plt.title(f"SHAP Dependence Plot: {top_feature} vs. Prediction", fontsize=14, fontname='Arial')
plt.show()
