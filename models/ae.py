import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras import layers, models

map_ruido = {1: 1.059, 2: 0.8, 3: 0.625}
print('opções: 1 (0.5); 2 (1.0); 3 (2.0)')
nivel = input("Informe o valor de epsilon (1, 2, 3 ou x): ")

X_ae = pd.read_csv("../data/ae_train.csv").values

if nivel.lower() == 'x':
    opt = 'adam'
else:
    nivel_int = int(nivel)
    noise_m = map_ruido.get(nivel_int, 0.625)
    opt = tfp.DPKerasAdamOptimizer(
        l2_norm_clip=1.3,
        noise_multiplier=noise_m,
        num_microbatches=1,
    )

def create_autoencoder(input_size, encoding_dim):
    input_layer = tf.keras.Input(shape=(input_size,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    # encoded = layers.Dense(64, activation='relu')(encoded)
    # encoded = layers.Dense(32, activation='relu')(encoded)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(encoded)
    # decoded = layers.Dense(64, activation='relu')(decoded)
    # decoded = layers.Dense(32, activation='relu')(decoded)
    decoded = layers.Dense(input_size, activation='sigmoid')(decoded)
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    return autoencoder

autoencoder = create_autoencoder(X_ae.shape[1], 16)
autoencoder.compile(optimizer=opt, loss='mse')
autoencoder.fit(X_ae, X_ae, epochs=10, batch_size=128, shuffle=True, validation_split=0.2)
