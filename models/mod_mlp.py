import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
X_train = pd.read_csv("data/X_train.csv").values
y_train = pd.read_csv("data/y_train.csv").values.flatten()

if nivel.lower() == 'x':
    opt = 'adam'
else:
    noise, clip = combinacoes.get(nivel, (0.625, 1.3))
    opt = tfp.DPKerasAdamOptimizer(
        l2_norm_clip=clip,
        noise_multiplier=noise,
        num_microbatches=1,
    )

model = Sequential()
model.add(Dense(190, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

