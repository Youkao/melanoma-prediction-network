import pickle
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard

NAME = f'melanoma-prediction-{int(time.time())}'

tensorboard = TensorBoard(log_dir=f'logs\\{NAME}\\')

X = pickle.load(open('X.pkl', 'rb'))
Y = pickle.load(open('Y.pkl', 'rb'))

X = X/255

model = Sequential()

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, input_shape = X.shape[1:], activation = 'relu'))

model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, epochs=20, validation_split=0.1, batch_size=32, callbacks=[tensorboard])
