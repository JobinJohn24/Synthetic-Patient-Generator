"""
Train GAN to generate synthetic patient data.
Optimized for distribution fidelity with advanced techniques.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


DATA_FILE = "patient_data.npy"
LATENT_DIM = 32
EPOCHS = 300
BATCH_SIZE = 16

X_real = np.load(DATA_FILE)
n_features = X_real.shape[1]

# Generator with BatchNorm for stability
generator = keras.Sequential([
    keras.layers.Input(shape=(LATENT_DIM,)),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(n_features, activation="sigmoid")
])

# Discriminator with LeakyReLU for better gradients
discriminator = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(512, activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(256, activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation="sigmoid")
])

# Optimizers with learning rate scheduling
d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)
g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

# Compile discriminator
discriminator.compile(optimizer=d_optimizer, loss="binary_crossentropy")

# Combined model
discriminator.trainable = False
gan_input = keras.layers.Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)
gan.compile(optimizer=g_optimizer, loss="binary_crossentropy")

# Training with improved strategy
for epoch in range(EPOCHS):
    # Train discriminator multiple times per generator update
    for _ in range(2):
        idx = np.random.randint(0, len(X_real), BATCH_SIZE)
        real_batch = X_real[idx]
        
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_batch = generator.predict(noise, verbose=0)
        
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_batch, np.ones((BATCH_SIZE, 1)) * 0.9)
        d_loss_fake = discriminator.train_on_batch(fake_batch, np.zeros((BATCH_SIZE, 1)) + 0.1)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
    
    # Train generator
    discriminator.trainable = False
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} | D Loss: {d_loss:.4f} | G Loss: {g_loss:.4f}")

# Save models
generator.save("generator.keras")
print("✓ Generator saved to generator.keras")
