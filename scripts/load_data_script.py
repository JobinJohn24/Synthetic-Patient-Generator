"""
Load and preprocess real patient data.
Data source: UCI Machine Learning - Heart Disease Dataset
https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import urllib.request

# === CUSTOMIZE HERE ===
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
OUTPUT_FILE = "patient_data.npy"
FEATURE_COUNT = 13
# ======================

# Load data
data = pd.read_csv(DATA_URL, header=None)
print(f"✓ Loaded {len(data)} patient records")

# Remove rows with missing values (denoted by '?')
data = data.replace('?', np.nan).dropna()
print(f"✓ Cleaned data: {len(data)} records")

# Select first FEATURE_COUNT columns (customize as needed)
X = data.iloc[:, :FEATURE_COUNT].values.astype(float)

# Normalize to [0, 1]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = (X_scaled - X_scaled.min()) / (X_scaled.max() - X_scaled.min())

# Save preprocessed data
np.save(OUTPUT_FILE, X_scaled)
print(f"✓ Saved preprocessed data to {OUTPUT_FILE}")
print(f"  Shape: {X_scaled.shape}")
