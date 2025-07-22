import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras import layers, models

noise_vals = [1.059, 0.8, 0.625]
clip_vals = [0.5, 1.3, 1.8]

combinacoes = {}
idx = 1
for noise in noise_vals:
    for clip in clip_vals:
        combinacoes[str(idx)] = (noise, clip)
        idx += 1

print("opções:")
for k, (noise, clip) in combinacoes.items():
    print(f"{k}: noise_multiplier={noise}, l2_norm_clip={clip}")
print("x: sem privacidade (adam)")

nivel = input("Informe o nível (1-9 ou x): ")
X_ae = pd.read_csv("data/ae_train.csv").values

if nivel.lower() == 'x':
    opt = 'adam'
else:
    noise, clip = combinacoes.get(nivel, (0.625, 1.3))
    opt = tfp.DPKerasAdamOptimizer(
        l2_norm_clip=clip,
        noise_multiplier=noise,
        num_microbatches=1,
    )

def create_autoencoder(input_size, encoding_dim):
    input_layer = tf.keras.Input(shape=(input_size,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
    decoded = layers.Dense(32, activation='relu')(encoded)
    decoded = layers.Dense(input_size, activation='sigmoid')(decoded)
    autoencoder = models.Model(inputs=input_layer, outputs=decoded)
    return autoencoder

autoencoder = create_autoencoder(X_ae.shape[1], 16)
autoencoder.compile(optimizer=opt, loss='mse')
autoencoder.fit(X_ae, X_ae, epochs=10, batch_size=128, shuffle=True, validation_split=0.2)

