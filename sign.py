import matplotlib.pyplot as plt
import sign_language

# Task 1
x_train, y_train, x_test, y_test = sign_language.load_data()

# Task 2
labels = ['A', 'B', 'C']

# Task 3
num_B_train = sum(y_train == 1)
num_C_train = sum(y_train == 2)
num_B_test = sum(y_test == 1)
num_C_test = sum(y_test == 2)

# Task 4
from keras.utils import to_categorical

y_train_OH = to_categorical(y_train)
y_test_OH = to_categorical(y_test)

# Task 5
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(10, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(15, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))

# Task 6
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Task 7
history = model.fit(x_train, y_train_OH, validation_split=0.2, epochs=2, batch_size=32)

# Task 8
x = x_test
y = y_test_OH
loss, accuracy = model.evaluate(x, y)

# Task 9
import numpy as np

y_probs = model.predict(x_test)
y_preds = np.argmax(y_probs, axis=1)
bad_test_idxs = np.where(y_preds != y_test)[0]
