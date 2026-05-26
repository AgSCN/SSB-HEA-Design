import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

# 1. Load Data
file_path = r'your_path_here\clean_data1.xlsx'
df = pd.read_excel(file_path)

# 2. Feature and Target Definition
target = 'formation_energy_per_atom'
features = [
    'volume_per_atom', 'mean MeltingT', 'mean CovalentRadius', 
    'range Electronegativity', 'mean Electronegativity', 
    'mean NpValence', 'mean NdValence', 'range Polarizability', 'mean Polarizability'
]

X = df[features]
y = df[target]

# 3. Split Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define Models
models = {
    "LGBM": LGBMRegressor(random_state=42, verbose=-1),
    "GBR": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel='rbf')
}

# 5. Calculation Loop
results = []
print(f"{'Model':<12} | {'Set':<8} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8}")
print("-" * 60)

for name, model in models.items():
    # 10-Fold Cross Validation (R2, MAE, RMSE)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # Changed 'neg_mean_squared_error' to 'neg_mean_absolute_error'
    cv_metrics = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
    cv_res = cross_validate(model, X_train_scaled, y_train, cv=kf, scoring=cv_metrics)
    
    cv_r2 = cv_res['test_r2'].mean()
    cv_mae = -cv_res['test_neg_mean_absolute_error'].mean()
    cv_rmse = -cv_res['test_neg_root_mean_squared_error'].mean()
    
    # Train Model
    model.fit(X_train_scaled, y_train)
    
    # Training Set Prediction
    y_train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Test Set Prediction
    y_test_pred = model.predict(X_test_scaled)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Print results to terminal
    print(f"{name:<12} | Train    | {train_r2:>8.4f} | {train_mae:>8.4f} | {train_rmse:>8.4f}")
    print(f"{'':<12} | CV (10f) | {cv_r2:>8.4f} | {cv_mae:>8.4f} | {cv_rmse:>8.4f}")
    print(f"{'':<12} | Test     | {test_r2:>8.4f} | {test_mae:>8.4f} | {test_rmse:>8.4f}")
    print("-" * 60)
    
    results.append({
        "Model": name,
        "Train_R2": train_r2, "CV_R2": cv_r2, "Test_R2": test_r2,
        "Train_MAE": train_mae, "CV_MAE": cv_mae, "Test_MAE": test_mae,
        "Train_RMSE": train_rmse, "CV_RMSE": cv_rmse, "Test_RMSE": test_rmse
    })

results_df = pd.DataFrame(results)

# 6. Plotting (Arial font & English)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
sns.set_theme(style="whitegrid")

# --- Figure 1: R² Comparison ---
plt.figure(figsize=(10, 6))
r2_plot = pd.melt(results_df, id_vars=['Model'], value_vars=['Train_R2', 'CV_R2', 'Test_R2'], var_name='Dataset', value_name='R2')
sns.barplot(x='Model', y='R2', hue='Dataset', data=r2_plot, palette='Blues_d')
plt.title(r'$R^2$ Comparison (Train, CV & Test)', fontsize=16)
plt.ylabel(r'$R^2$ Score')
plt.ylim(results_df['CV_R2'].min() * 0.98, 1.02)
plt.show()

# --- Figure 2: MAE Comparison (Replaced MSE) ---
plt.figure(figsize=(10, 6))
mae_plot = pd.melt(results_df, id_vars=['Model'], value_vars=['Train_MAE', 'CV_MAE', 'Test_MAE'], var_name='Dataset', value_name='MAE')
sns.barplot(x='Model', y='MAE', hue='Dataset', data=mae_plot, palette='Oranges_d')
plt.title('MAE Comparison', fontsize=16)
plt.ylabel('Mean Absolute Error (eV/atom)')
plt.show()

# --- Figure 3: RMSE Comparison ---
plt.figure(figsize=(10, 6))
rmse_plot = pd.melt(results_df, id_vars=['Model'], value_vars=['Train_RMSE', 'CV_RMSE', 'Test_RMSE'], var_name='Dataset', value_name='RMSE')
sns.barplot(x='Model', y='RMSE', hue='Dataset', data=rmse_plot, palette='Greens_d')
plt.title('RMSE Comparison', fontsize=16)
plt.ylabel('Root Mean Squared Error (eV/atom)')
plt.show()

# --- Figure 4: Multi-metric Algorithm Comparison (Test Set) ---
plt.figure(figsize=(12, 7))
test_metrics = results_df[['Model', 'Test_R2', 'Test_MAE', 'Test_RMSE']]
# Renaming for clean legend
test_metrics.columns = ['Model', r'$R^2$', 'MAE', 'RMSE']
metrics_melted = pd.melt(test_metrics, id_vars=['Model'], var_name='Metric', value_name='Value')

sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_melted, palette='muted')
plt.title(r'Test Set Performance Comparison ($R^2$, MAE, RMSE)', fontsize=16)
plt.xlabel('Machine Learning Algorithms')
plt.ylabel('Value')
# Note: Log scale is removed here because MAE/RMSE/R2 are often in a similar range (0.01 to 1.0)
# But you can add plt.yscale('log') if the columns are too short to see.
plt.show()
