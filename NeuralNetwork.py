import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

train_data=np.asarray([[5,6,7,1,35,34], [5,6,9,2,30,21],[51,1,3,4,5,0.5]])
train_labels=np.asarray([0,1,0]).reshape(3,1)

model=Sequential()
model.add(Dense(256, input_dim=len(train_data[0]), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="softmax"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_data, train_labels, batch_size=16, epochs=20, validation_split=0.1, verbose=0)
predictions = model.predict(train_data, batch_size=16, verbose=0)
print(predictions)
