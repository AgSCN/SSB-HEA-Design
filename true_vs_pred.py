import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from lightgbm import LGBMRegressor


file_path = r'your_path_here\clean_data1.csv'
df = pd.read_csv(file_path)


target = 'formation_energy_per_atom'
features = [col for col in df.columns if col not in [target, 'formula_pretty']]

X = df[features]
y = df[target]
formulas = df['formula_pretty']

X_train, X_test, y_train, y_test, formulas_train, formulas_test = train_test_split(
    X, y, formulas, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lgbm = LGBMRegressor(random_state=42, verbose=-1)
lgbm.fit(X_train_scaled, y_train)

y_train_pred = lgbm.predict(X_train_scaled)
y_test_pred = lgbm.predict(X_test_scaled)

def plot_regression(y_true, y_pred, title, color):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color=color, label='Data Points')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.annotate(f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.4f}', 
                 xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title(title, fontsize=15)
    plt.xlabel('Actual Formation Energy (eV/atom)', fontsize=12)
    plt.ylabel('Predicted Formation Energy (eV/atom)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_regression(y_train, y_train_pred, 'LGBM: Training Set (Actual vs Predicted)', 'royalblue')
plot_regression(y_test, y_test_pred, 'LGBM: Test Set (Actual vs Predicted)', 'seagreen')

output_dir = r'your_path_here\\'

train_output = pd.DataFrame({
    'Formula': formulas_train,
    'Actual': y_train,
    'Predicted': y_train_pred
})

test_output = pd.DataFrame({
    'Formula': formulas_test,
    'Actual': y_test,
    'Predicted': y_test_pred
})

train_output.to_csv(output_dir + 'LGBM_Training_Results.csv', index=False)
test_output.to_csv(output_dir + 'LGBM_Test_Results.csv', index=False)

print(f"File successfully exported to {output_dir}")
