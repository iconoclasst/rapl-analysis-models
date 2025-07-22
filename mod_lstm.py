import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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

X_train_2d = pd.read_csv("../data/lstmx_train.csv").values
X_train = X_train_2d.reshape(X_train_2d.shape[0], 1, X_train_2d.shape[1])

y_train_1d = pd.read_csv("../data/lstmy_train.csv").values.flatten()
y_train = to_categorical(y_train_1d)
num_classes = y_train.shape[1]

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
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

