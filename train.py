import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.callbacks import ReduceLROnPlateau

text = open('C:/Users/meruy/Desktop/sonnets.txt', encoding='utf8').read().lower()

chars = set(text)
sorted_chars = sorted(chars)
char_index=dict()
for index, char in enumerate(sorted_chars):
    char_index.update({char:index})
reverse_char_index = dict([(value, key) for (key, value) in char_index.items()])

seq_size = 50
step = 1

sentences = []
target = []

for i in range(0, len(text)-seq_size, step):
    sentences.append(text[i:i+seq_size])
    target.append(text[i+seq_size])

x = np.zeros((len(sentences), seq_size, len(sorted_chars)))
y = np.zeros((len(sentences), len(sorted_chars)))
for i, sentence in enumerate(sentences):
    for j, char in enumerate(sentence):
        x[i, j, char_index[char]] = 1
    y[i, char_index[target[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape = (seq_size, len(sorted_chars))))
model.add(Dense(len(sorted_chars)))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(lr= 0.01)) #HOW TO EVALUATE THIS

new_model = "weights.h5"
print_callback = LambdaCallback()
checkpoint = ModelCheckpoint(new_model, monitor='loss',verbose=1, save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=1, min_lr=0.001)
callbacks = [print_callback, checkpoint, reduce_lr]

model.fit(x, y, batch_size=128, epochs=1, callbacks=callbacks)
