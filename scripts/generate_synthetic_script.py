"""
Generate synthetic patient data using trained GAN.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

#  MAKE ADJUSTMENT HERE # 
GENERATOR_FILE = "generator.keras"
NUM_SYNTHETIC = 500
LATENT_DIM = 32
OUTPUT_FILE = "synthetic_patients.npy"

# Load generator
generator = keras.models.load_model(GENERATOR_FILE)
print("✓ Loaded generator model")

# Generate synthetic data
noise = np.random.normal(0, 1, (NUM_SYNTHETIC, LATENT_DIM))
synthetic_data = generator.predict(noise)

# Clip to valid range [0, 1]
synthetic_data = np.clip(synthetic_data, 0, 1)

# Save
np.save(OUTPUT_FILE, synthetic_data)
print(f"✓ Generated {NUM_SYNTHETIC} synthetic patient records")
print(f"  Saved to {OUTPUT_FILE}")
print(f"  Shape: {synthetic_data.shape}")
