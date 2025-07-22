import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

#niveis            = [nv1, nv2, nv3    ]
#niveis de epsilon = [0.5, 1.0, 2.0    ]
#niveis de ruido   = [1.059, 0.8, 0.625]

map_ruido = {1: 1.059, 2: 0.8, 3: 0.625}
print('opções: 1 (0.5); 2 (1.0); 3 (2.0)')
nivel = input("Informe o valor de epsilon (1, 2, 3 ou x): ")

X_train_2d = pd.read_csv("../data/lstmx_train.csv").values
X_train = X_train_2d.reshape(X_train_2d.shape[0], 1, X_train_2d.shape[1])

y_train_1d = pd.read_csv("../data/lstmy_train.csv").values.flatten()
y_train = to_categorical(y_train_1d)
num_classes = y_train.shape[1]

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
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
