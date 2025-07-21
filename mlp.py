import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras import Sequential, Dense

map_ruido = {1: 1.059, 2: 0.8, 3: 0.625}
print('opções: 1 (0.5); 2 (1.0); 3 (2.0)')
nivel = input("Informe o valor de epsilon (1, 2, 3 ou x): ")

X_train = pd.read_csv("data/X_train.csv").values
y_train = pd.read_csv("data/y_train.csv").values.flatten()

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

model = Sequential()
model.add(Dense(190, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
