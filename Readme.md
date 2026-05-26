# Materials Formation Energy Prediction & HEA Screening (Materials Machine Learning Project)

This project is an integrated machine learning workflow for materials science. It leverages a Python-based toolchain to predict the **Formation Energy** of materials related to solid-state batteries (such as LLZO interface modification materials) and performs composition optimization and screening for **High-Entropy Alloys (HEA)** based on predictive insights.

## Core Features

* **Multidimensional Feature Extraction**: Automatically extracts physicochemical descriptors (e.g., Magpie and Oliynyk sets) from chemical formulas using `matminer` and `pymatgen`.
* **Feature Engineering & Dimensionality Reduction**: Optimizes model input by eliminating redundant features through **Mutual Information (MI)** and **Pearson Correlation** analysis.
* **Multi-Algorithm Benchmarking**: Supports performance comparison across multiple models, including **LightGBM**, **Gradient Boosting (GBR)**, and **SVR**, integrated with **10-fold cross-validation**.
* **Model Interpretability**: Incorporates **SHAP (SHapley Additive exPlanations)** analysis to reveal the contributions of physical properties like polarizability and electronegativity to material stability.
* **Iterative HEA Screening**: Supports formation energy prediction and stability ranking for complex systems containing five or more elements.

## 1. Data Preparation & Feature Extraction

* `oliynyk_features.py`: Batch extracts physicochemical features from `formula_pretty.csv`.
* `polarizability_feature.py`: Specifically extracts statistical features related to **Polarizability** for interface stability research.

## 2. Feature Selection & Preprocessing

* `correlation_analysis.py`: Computes correlation heatmaps, utilizes Mutual Information to prune irrelevant variables, and handles multicollinearity among features.

## 3. Model Construction & Evaluation

* `model.py`: Implements benchmarking for LGBM, GBR, and SVR; outputs $R^2$, MAE, and RMSE metrics; and generates algorithm performance comparison charts.
* `true_vs_pred.py`: Plots "Actual vs. Predicted" regression scatter plots for both training and testing sets, exporting results to CSV for further analysis.

## 4. Model Interpretation & Material Screening

* `shap_plot.py`: Generates SHAP summary plots, bar charts, and dependence plots to interpret the model from a physical perspective.
* `single_mental_prediction.py`: Evaluates stability and provides rankings for binary systems (Li-M).
* `hea_prediction.py`: Performs exhaustive combinatorial iterations for **5-component HEA systems**, screening for the optimal composition with the lowest average formation energy.