import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp

map_ruido = {1: 1.059, 2: 0.8, 3: 0.625}
nivel = int(input("Informe o nivel de ruido (1, 2 ou 3): "))
noise_m = map_ruido.get(nivel, 0.625)

X_train = pd.read_csv("X_train.csv").values
y_train = pd.read_csv("y_train.csv").values

opt = tfp.privacy.optimizers.dp_optimizer_keras.DPKerasAdamOptimizer(
    l2_norm_clip=1.3,
    noise_multiplier=noise_m,
    num_microbatches=1,
)

from tensorflow.keras import Sequential, Dense

model = Sequential()
model.add(Dense(190, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
